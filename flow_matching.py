"""
Flow Matching with PyTorch. Has continuous time embedding.
Everything in one file.
Same interface as diffusion.py but uses Flow Matching algorithm.

Flow Matching trains a velocity field v(x, t) that transports
noise (t=0) to data (t=1) via the ODE: dx/dt = v(x, t).
Conditional Flow Matching uses linear paths:
  x_t = (1 - t) * x0 + t * x1   (x0=noise, x1=data)
  target velocity: v = x1 - x0   (constant along each path)
"""

##################################################################################################################################
import math
import torch
import torch.nn as nn


class FlowMatcher:
    """
    Linear conditional flow matching.
    Interpolates between noise x0 and data x1 along straight paths.
    """
    def interpolate(self, x1, t):
        # x1: (N, C, H, W), t: (N,) in [0, 1]
        n = len(x1)
        x0 = torch.randn_like(x1)
        t_view = t.reshape(n, 1, 1, 1)
        xt = (1 - t_view) * x0 + t_view * x1
        velocity = x1 - x0
        return xt, velocity

    def interpolate_1d(self, x1, t):
        # x1: (N, C), t: (N,) in [0, 1]
        n = len(x1)
        x0 = torch.randn_like(x1)
        t_view = t.reshape(n, 1)
        xt = (1 - t_view) * x0 + t_view * x1
        velocity = x1 - x0
        return xt, velocity


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t in [0, 1]."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        # t: (N,) or (N, 1)
        if t.dim() > 1:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (N, dim)
        return self.proj(emb)


class ConvBlock(nn.Module):
    def __init__(self, in_shape, out_c, kernel_size=3, stride=1, padding=1, normalize=True):
        super(ConvBlock, self).__init__()
        self.ln = nn.LayerNorm(in_shape)
        self.conv1 = nn.Conv2d(in_shape[0], out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU()
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class VelocityNet(nn.Module):
    """
    UNet that predicts the velocity field v(x, t) for flow matching.
    Architecture mirrors diffusion.py UNet; time embedding handles continuous t.
    """
    def __init__(self, time_emb_dim=100):
        super(VelocityNet, self).__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

        # First half
        self.te1 = nn.Linear(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            ConvBlock((1, 28, 28), 10),
            ConvBlock((10, 28, 28), 10),
            ConvBlock((10, 28, 28), 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = nn.Linear(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            ConvBlock((10, 14, 14), 20),
            ConvBlock((20, 14, 14), 20),
            ConvBlock((20, 14, 14), 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = nn.Linear(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            ConvBlock((20, 7, 7), 40),
            ConvBlock((40, 7, 7), 40),
            ConvBlock((40, 7, 7), 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = nn.Linear(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            ConvBlock((40, 3, 3), 20),
            ConvBlock((20, 3, 3), 20),
            ConvBlock((20, 3, 3), 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = nn.Linear(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            ConvBlock((80, 7, 7), 40),
            ConvBlock((40, 7, 7), 20),
            ConvBlock((20, 7, 7), 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = nn.Linear(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            ConvBlock((40, 14, 14), 20),
            ConvBlock((20, 14, 14), 10),
            ConvBlock((10, 14, 14), 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = nn.Linear(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            ConvBlock((20, 28, 28), 10),
            ConvBlock((10, 28, 28), 10),
            ConvBlock((10, 28, 28), 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x: (N, 1, 28, 28), t: (N,) or (N, 1) in [0, 1]
        t = self.time_embed(t)  # (N, time_emb_dim)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up3(out5)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        return self.conv_out(out)


##################################################################################################################################
import torchvision
from tqdm.auto import tqdm


def get_dataloader(batch_size=128):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.mnist.MNIST("./data", download=True, train=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def get_device():
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps:0'
    if torch.cuda.is_available():
        device = 'cuda:0'
    return device


def train(n_epochs, batch_size=128, time_emb_dim=100, model_path='flow_matching.pth'):
    device = get_device()
    dataloader = get_dataloader(batch_size=batch_size)
    flow_matcher = FlowMatcher()
    net = VelocityNet(time_emb_dim=time_emb_dim).to(device)
    optim = torch.optim.Adam(net.parameters())

    net.train()
    with tqdm(range(n_epochs), colour="#00ee00") as epoch_pbar:
        for _ in epoch_pbar:
            with tqdm(dataloader, leave=False, colour="#005500") as batch_pbar:
                for images, _ in batch_pbar:
                    x1 = images.to(device)
                    n = len(x1)
                    t = torch.rand(n).to(device)  # uniform t in [0, 1]
                    xt, velocity = flow_matcher.interpolate(x1, t)
                    velocity_hat = net(xt, t)
                    loss = nn.functional.mse_loss(velocity_hat, velocity)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    batch_pbar.set_description(f'{loss.item():.3f}')
    torch.save(net.state_dict(), model_path)


##################################################################################################################################
import matplotlib.pyplot as plt


def show_images(images):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(4, 4))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < len(images):
                fig.add_subplot(rows, cols, idx + 1)
                plt.imshow(images[idx][0], cmap="gray")
                plt.axis('off')
                idx += 1
    plt.tight_layout()
    plt.show()


def predict(n_samples=16, c=1, h=28, w=28, n_steps=100, time_emb_dim=100, model_path='flow_matching.pth'):
    """Euler ODE integration from t=0 (noise) to t=1 (data)."""
    device = get_device()
    net = VelocityNet(time_emb_dim=time_emb_dim).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    net.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, c, h, w).to(device)
        dt = 1.0 / n_steps
        for i in tqdm(range(n_steps)):
            t = torch.full((n_samples,), i * dt, device=device)
            velocity = net(x, t)
            x = x + dt * velocity
    show_images(x)


def predict_fast(fast_steps=10, n_samples=16, c=1, h=28, w=28, time_emb_dim=100, model_path='flow_matching.pth'):
    """
    Fast generation with fewer ODE steps (analogous to predict_ddim).
    Flow matching naturally supports very few steps without quality collapse.
    """
    predict(n_samples=n_samples, c=c, h=h, w=w, n_steps=fast_steps,
            time_emb_dim=time_emb_dim, model_path=model_path)


##################################################################################################################################
from absl import flags
from absl import app


def main(unused_args):
    """
    Samples:
      python flow_matching.py --train --epochs 100 --predict --fast
    """
    if FLAGS.train:
        train(n_epochs=FLAGS.epochs)

    if FLAGS.predict:
        if FLAGS.fast:
            predict_fast()
        else:
            predict()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 3, "Epochs to train")
    flags.DEFINE_bool("fast", False, "Faster generation with fewer ODE steps")

    app.run(main)
