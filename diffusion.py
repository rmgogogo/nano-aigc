"""
Diffusion with PyTorch. Has time embedding.
Everything in one file.
"""

##################################################################################################################################
import torch
import torch.nn as nn

class Noiser:
    '''
    Noiser generates the noise. It's the diffusion process.
    Given x0 is the orignal image, it can geneate x1, x2, ..., xt images which has more noise in steps.
    '''
    def __init__(self, device, n_steps=1000, beta_min=0.0001, beta_max=0.02):
        self.device = device
        self.n_steps = 1000
        self.betas = torch.linspace(beta_min, beta_max, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def noisy(self, x0, t):
        with torch.no_grad():
            n, c, h, w = x0.shape
            alpha_bar = self.alpha_bars[t]
            epislon = torch.randn(n, c, h, w).to(self.device)
            xt = alpha_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - alpha_bar).sqrt().reshape(n, 1, 1, 1) * epislon
            return xt, epislon

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

class UNet(nn.Module):
    '''
    UNet is used to predict the noise, given an image with noise, it predict the noise part.
    '''
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        # Time Embedding, type of positional embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)

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
        # x is (N, 1, 28, 28)
        t = self.time_embed(t) # (N, time-embedding-features)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))   # (N, 1, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 12, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)
        return out


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
        device = 'cuda'
    return device

def train(n_epochs, batch_size=128, n_steps=1000, beta_min=0.0001, beta_max=0.02, time_emb_dim=100, model_path='diffusion.pth'):
    device = get_device()
    dataloader = get_dataloader(batch_size=batch_size)
    noiser = Noiser(device=device, n_steps=n_steps, beta_min=beta_min, beta_max=beta_max)
    net = UNet(n_steps=n_steps, time_emb_dim=time_emb_dim).to(device)
    optim = torch.optim.Adam(net.parameters())
    
    net.train()
    with tqdm(range(n_epochs), colour="#00ee00") as epoch_pbar:
        for _ in epoch_pbar:
            with tqdm(dataloader, leave=False, colour="#005500") as batch_pbar:
                for images, _ in batch_pbar:
                    # generate (xt, epsilon) pair for training, it also can be implemented as torchvision.transforms
                    x0 = images.to(device)
                    x0_batch = len(x0)
                    t = torch.randint(0, n_steps, (x0_batch,)).to(device)
                    xt, epsilon = noiser.noisy(x0, t)
                    epsilon_hat = net(xt, t.reshape(x0_batch, -1))
                    loss = nn.functional.mse_loss(epsilon_hat, epsilon)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    batch_pbar.set_description(f'{loss.item():.3f}')
    torch.save(net.state_dict(), model_path)

##################################################################################################################################
import matplotlib.pyplot as plt

def show_images(images):
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(4, 4))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
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

def predict(n_samples=16, c=1, h=28, w=28, n_steps=1000, beta_min=0.0001, beta_max=0.02, time_emb_dim=100, model_path='diffusion.pth'):
    device = get_device()
    noiser = Noiser(device=device, n_steps=n_steps, beta_min=beta_min, beta_max=beta_max)
    net = UNet(n_steps=n_steps, time_emb_dim=time_emb_dim).to(device)
    net.load_state_dict(torch.load(model_path))

    net.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, c, h, w).to(device)
        for _, t in enumerate(list(range(n_steps))[::-1]): # 999->0
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            epislon = net(x, time_tensor)
            alpha_t = noiser.alphas[t]
            alpha_t_bar = noiser.alpha_bars[t]
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * epislon)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                beta_t = noiser.betas[t]
                x = x + beta_t.sqrt() * z
    show_images(x)

##################################################################################################################################

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_bool("train", False, "Train the model")
flags.DEFINE_bool("predict", False, "Predict")
flags.DEFINE_integer("epochs", 3, "Epochs to train")

def main(unused_args):
    """
    Samples:
      python diffusion.py --train --epochs 5 --predict
    """
    if FLAGS.train:
        train(n_epochs=FLAGS.epochs)

    if FLAGS.predict:
        predict()

if __name__ == '__main__':
    app.run(main)