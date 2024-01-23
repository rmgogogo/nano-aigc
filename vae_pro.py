"""
VAE with PyTorch. Compared with vae.py, it uses conv2d.
Everything in one file.
"""

##################################################################################################################################
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2, 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
        )
        self.calc_mean = nn.Linear(16, latent)
        self.calc_logvar = nn.Linear(16, latent)
    
    def forward(self, x):
        x = self.encode(x) # [N, 16, 1, 1]
        x = x.view(x.shape[0], -1) # [N, 16]
        return self.calc_mean(x), self.calc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent):
        super(Decoder, self).__init__()
        self.map_back = nn.Linear(latent, 16)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        # x: [N, latent]
        x = self.map_back(x) # [N, 16]
        x = x.view(-1, 16, 1, 1)
        x = self.decode(x) # [N, 1, 28, 28]
        return x
    
class VAE(nn.Module):
    def __init__(self, latent):
        super(VAE, self).__init__()
        self.latent = latent
        self.encoder = Encoder(latent)
        self.decoder = Decoder(latent)
        
    def sampling(self, mean, logvar):
        sample = torch.randn(mean.shape).to(mean.device)
        stdvar = torch.exp(0.5 * logvar)
        return mean + sample * stdvar
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar
    
    def generate(self, batch_size = 1):
        model_device = next(self.parameters()).device
        z = torch.randn((batch_size, self.latent)).to(model_device)
        return self.decoder(z)

##################################################################################################################################
import torchvision
import tqdm

def get_dataloader():
    tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)

def get_device():
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps:0'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def loss(x_target, x_actual, mean, logvar):
    reconstruction_loss = nn.functional.mse_loss(x_actual, x_target, reduction='sum')
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence

def train(net, dataloader, device):
    optimizer = torch.optim.AdamW(net.parameters())
    net.train()
    with tqdm.tqdm(dataloader, ncols=64) as pbar:
        for x, _ in pbar:
            x = x.to(device)
            x_actual, mean, logvar = net(x)
            l = loss(x, x_actual, mean, logvar).to(device)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            pbar.set_description(f"Loss {l.cpu().item():.4f}")

##################################################################################################################################

from matplotlib import pyplot as plt

def predict(net):
    net.eval() # disable drop-out and batch-normalization
    with torch.no_grad():
        x = net.generate(16)

    images = x.cpu()
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(0).numpy(), cmap='gray')
        ax.axis("off")
    plt.tight_layout()
    plt.show()

##################################################################################################################################

from absl import flags
from absl import app

def main(unused_args):
    """
    Samples:
      python vae_pro.py --train --predict
    """

    device = get_device()
    if FLAGS.train:
        print('Train')
        dataloader = get_dataloader()
        net = VAE(latent=FLAGS.latent).to(device)
        for i in range(FLAGS.epochs):
            train(net, dataloader, device)
        torch.save(net.state_dict(), 'vae_pro.pth')

    if FLAGS.predict:
        print('Predict')
        net = VAE(latent=FLAGS.latent).to(device)
        net.load_state_dict(torch.load('vae_pro.pth'))
        predict(net)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 3, "Epochs to train")
    flags.DEFINE_integer("latent", 8, "Epochs to train")

    app.run(main)