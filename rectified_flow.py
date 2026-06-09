"""
Rectified Flow with PyTorch.
Everything in one file.

Rectified Flow (Liu et al. 2022) builds on Flow Matching with one key addition — Reflow:
  Round 1: Train with independent coupling (identical to Flow Matching)
  Reflow:  Run the trained ODE on noise samples to collect causally-coupled (x0, x1) pairs,
           then retrain on those pairs. Coupled paths don't cross → straighter velocity field
           → fewer ODE steps needed at inference.

Usage:
  python rectified_flow.py --train --epochs 100        # Round 1
  python rectified_flow.py --reflow --epochs 100       # Reflow → Round 2
  python rectified_flow.py --predict                   # Generate (uses r2 if available, else r1)
  python rectified_flow.py --predict --fast            # Fewer ODE steps
"""

##################################################################################################################################
import os
import math
import torch
import torch.nn as nn


class FlowMatcher:
    """Linear interpolation between noise (x0) and data (x1)."""

    def interpolate(self, x1, t):
        # Round 1: x0 sampled independently — same as Flow Matching.
        # x1: (N, C, H, W), t: (N,) in [0, 1]
        n = len(x1)
        x0 = torch.randn_like(x1)
        t_view = t.reshape(n, 1, 1, 1)
        xt = (1 - t_view) * x0 + t_view * x1
        return xt, x1 - x0, x0

    def interpolate_coupled(self, x0, x1, t):
        # Reflow: x0 and x1 are causally paired — paths won't cross.
        # x0, x1: (N, C, H, W), t: (N,) in [0, 1]
        n = len(x1)
        t_view = t.reshape(n, 1, 1, 1)
        xt = (1 - t_view) * x0 + t_view * x1
        return xt, x1 - x0


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t in [0, 1]."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        if t.dim() > 1:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
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
    """UNet velocity field predictor — identical architecture to flow_matching.py."""
    def __init__(self, time_emb_dim=100):
        super(VelocityNet, self).__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

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

        self.te_mid = nn.Linear(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            ConvBlock((40, 3, 3), 20),
            ConvBlock((20, 3, 3), 20),
            ConvBlock((20, 3, 3), 40)
        )

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
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))
        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))
        out4 = self.b4(torch.cat((out3, self.up1(out_mid)), dim=1) + self.te4(t).reshape(n, -1, 1, 1))
        out5 = self.b5(torch.cat((out2, self.up2(out4)), dim=1) + self.te5(t).reshape(n, -1, 1, 1))
        out = self.b_out(torch.cat((out1, self.up3(out5)), dim=1) + self.te_out(t).reshape(n, -1, 1, 1))
        return self.conv_out(out)


class CoupledDataset(torch.utils.data.Dataset):
    """Holds causally-coupled (x0, x1) pairs produced by reflow."""
    def __init__(self, x0, x1):
        self.x0 = x0  # (N, 1, 28, 28) on CPU
        self.x1 = x1

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx]


##################################################################################################################################
import torchvision
from tqdm.auto import tqdm

R1_PATH = 'rectified_flow_r1.pth'
R2_PATH = 'rectified_flow_r2.pth'


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


def _euler(net, x, n_steps, device):
    """Shared Euler ODE integration loop."""
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((len(x),), i * dt, device=device)
        x = x + dt * net(x, t)
    return x


def train(n_epochs, batch_size=128, time_emb_dim=100, model_path=R1_PATH):
    """Round 1: independent coupling, identical to Flow Matching."""
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
                    t = torch.rand(n, device=device)
                    xt, velocity, _ = flow_matcher.interpolate(x1, t)
                    loss = nn.functional.mse_loss(net(xt, t), velocity)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    batch_pbar.set_description(f'{loss.item():.3f}')
    torch.save(net.state_dict(), model_path)


def generate_reflow_pairs(n_samples=60000, n_steps=100, batch_size=256,
                          time_emb_dim=100, r1_path=R1_PATH):
    """
    Run the Round-1 ODE on noise to collect causally-coupled (x0, x1) pairs.
    Each x0 deterministically maps to one x1, so their connecting paths never cross.
    """
    device = get_device()
    net = VelocityNet(time_emb_dim=time_emb_dim).to(device)
    net.load_state_dict(torch.load(r1_path, map_location=device))
    net.eval()

    all_x0, all_x1 = [], []
    n_done = 0
    print(f'Generating {n_samples} reflow pairs...')
    with torch.no_grad():
        with tqdm(total=n_samples, colour="#0055ee") as pbar:
            while n_done < n_samples:
                bs = min(batch_size, n_samples - n_done)
                x0 = torch.randn(bs, 1, 28, 28, device=device)
                x1 = _euler(net, x0.clone(), n_steps, device)
                all_x0.append(x0.cpu())
                all_x1.append(x1.cpu())
                n_done += bs
                pbar.update(bs)

    return torch.cat(all_x0), torch.cat(all_x1)


def reflow(n_epochs, batch_size=128, n_reflow_samples=60000, ode_steps=100,
           time_emb_dim=100, r1_path=R1_PATH, model_path=R2_PATH):
    """
    Round 2: generate coupled pairs via the Round-1 ODE, then retrain.
    The new model learns a straighter velocity field that needs fewer inference steps.
    """
    device = get_device()
    flow_matcher = FlowMatcher()

    x0_data, x1_data = generate_reflow_pairs(
        n_samples=n_reflow_samples, n_steps=ode_steps,
        batch_size=256, time_emb_dim=time_emb_dim, r1_path=r1_path
    )
    dataset = CoupledDataset(x0_data, x1_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    net = VelocityNet(time_emb_dim=time_emb_dim).to(device)
    optim = torch.optim.Adam(net.parameters())

    net.train()
    with tqdm(range(n_epochs), colour="#00ee00") as epoch_pbar:
        for _ in epoch_pbar:
            with tqdm(dataloader, leave=False, colour="#005500") as batch_pbar:
                for x0, x1 in batch_pbar:
                    x0, x1 = x0.to(device), x1.to(device)
                    n = len(x1)
                    t = torch.rand(n, device=device)
                    xt, velocity = flow_matcher.interpolate_coupled(x0, x1, t)
                    loss = nn.functional.mse_loss(net(xt, t), velocity)
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


def predict(n_samples=16, c=1, h=28, w=28, n_steps=100, time_emb_dim=100):
    """Euler ODE integration. Prefers Round-2 model; falls back to Round-1."""
    model_path = R2_PATH if os.path.exists(R2_PATH) else R1_PATH
    print(f'Using {model_path}')
    device = get_device()
    net = VelocityNet(time_emb_dim=time_emb_dim).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    net.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, c, h, w, device=device)
        x = _euler(net, x, n_steps, device)
    show_images(x)


def predict_fast(fast_steps=10, n_samples=16, c=1, h=28, w=28, time_emb_dim=100):
    """
    Few-step generation. After reflow the velocity field is much straighter,
    so 1-10 steps often give good results — the main payoff of Rectified Flow.
    """
    predict(n_samples=n_samples, c=c, h=h, w=w, n_steps=fast_steps, time_emb_dim=time_emb_dim)


##################################################################################################################################
from absl import flags
from absl import app


def main(unused_args):
    """
    Samples:
      python rectified_flow.py --train --epochs 100
      python rectified_flow.py --reflow --epochs 100
      python rectified_flow.py --predict --fast
    """
    if FLAGS.train:
        train(n_epochs=FLAGS.epochs)

    if FLAGS.reflow:
        reflow(n_epochs=FLAGS.epochs)

    if FLAGS.predict:
        if FLAGS.fast:
            predict_fast()
        else:
            predict()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Round 1 training with independent coupling")
    flags.DEFINE_bool("reflow", False, "Generate coupled pairs and train Round 2")
    flags.DEFINE_bool("predict", False, "Generate images")
    flags.DEFINE_integer("epochs", 3, "Epochs to train")
    flags.DEFINE_bool("fast", False, "Fewer ODE steps (payoff is larger after reflow)")

    app.run(main)
