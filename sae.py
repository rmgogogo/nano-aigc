"""
Sparse AutoEncoder with PyTorch.
Everything in one file.
"""

##################################################################################################################################
import torch
import torch.nn as nn

def decorrelation_loss(encoded_batch):
    # encoded_batch: shape (B, D)
    encoded_centered = encoded_batch - encoded_batch.mean(dim=0, keepdim=True)
    cov = (encoded_centered.T @ encoded_centered) / encoded_batch.size(0)  # D x D
    identity = torch.eye(cov.size(0), device=cov.device)
    return ((cov - identity) ** 2).sum()

class Encoder(nn.Module):
    def __init__(self, latent):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(28 * 28, latent),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encode(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(latent, 28 * 28),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decode(x)
    
class SAE(nn.Module):
    def __init__(self, latent):
        super(SAE, self).__init__()
        self.latent = latent
        self.encoder = Encoder(latent)
        self.decoder = Decoder(latent)
        
    def forward(self, x):
        x = x.contiguous().view(-1, 28 * 28)
        z = self.encoder(x)
        x = self.decoder(z)
        x = x.contiguous().view(-1, 1, 28, 28)
        return x, z

##################################################################################################################################
import torchvision
import tqdm

def get_dataloader(train=True, batch_size=128):
    tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        "./data",
        train=train,
        download=True,
        transform=tf,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

def get_device():
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps:0'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def loss(x_target, x_actual, z):
    reconstruction_loss = nn.functional.mse_loss(x_actual, x_target, reduction='mean')
    l1_loss = torch.mean(torch.abs(z))
    l_diverse = decorrelation_loss(z)
    return reconstruction_loss + l1_loss * 0.12 + l_diverse * 0.1

def train(net, dataloader, device):
    optimizer = torch.optim.AdamW(net.parameters())
    net.train()
    with tqdm.tqdm(dataloader, ncols=64) as pbar:
        for x, _ in pbar:
            x = x.to(device)
            x_actual, z = net(x)
            l = loss(x, x_actual, z).to(device)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            pbar.set_description(f"Loss {l.cpu().item():.4f}")

##################################################################################################################################

from matplotlib import pyplot as plt

def predict(net, dataloader, device):
    net.eval() # disable drop-out and batch-normalization
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        output, z = net(images)

    encoded_flat = z.cpu()
    threshold = 0 #1e-5
    non_zero = (encoded_flat.abs() > threshold).sum().item()
    total = encoded_flat.numel()
    active_ratio = round(non_zero * 100 / total, 2)
    print(f"Activate Ratio: {active_ratio}%")

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0][i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0][i].axis('off')
        axes[1][i].imshow(output[i].cpu().view(28, 28), cmap='gray')
        axes[1][i].axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.imshow(encoded_flat, aspect='auto', cmap='hot')
    plt.colorbar(label='Activation Value')
    plt.tight_layout()
    plt.show()

##################################################################################################################################

from absl import flags
from absl import app

def main(unused_args):
    """
    Samples:
      python ae.py --train --epochs 3 --predict
    """

    device = get_device()
    if FLAGS.train:
        print('Train')
        dataloader = get_dataloader()
        net = SAE(latent=128).to(device)
        for i in range(FLAGS.epochs):
            train(net, dataloader, device)
        torch.save(net.state_dict(), 'sae.pth')

    if FLAGS.predict:
        print('Predict')
        dataloader = get_dataloader(train=False, batch_size=16)
        net = SAE(latent=128).to(device)
        net.load_state_dict(torch.load('sae.pth'))
        predict(net, dataloader, device)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 3, "Epochs to train")

    app.run(main)