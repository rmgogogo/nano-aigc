"""
VQ-VAE with PyTorch.
Everything in one file.
"""

##################################################################################################################################
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape
    def forward(self, x):
        return x.view(*self.out_shape)

class Encoder(nn.Module):
    def __init__(self, latent):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(128, latent)
        )
    def forward(self, x):
        return self.encode(x)

class Decoder(nn.Module):
    def __init__(self, latent):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(in_features=latent, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=28*28),
            nn.Sigmoid(),
            Reshape((-1, 1, 28, 28))
        )
    def forward(self, x):
        return self.decode(x)
    
# Vector Quantization Layer
class VectorQuantization(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super(VectorQuantization, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # x: [N, embedding_dim]
        # Find nearest embedding, (a-b)**2=a**2+b**2-2ab
        distances = torch.sum(x**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - \
            2 * torch.matmul(x, self.embedding.weight.t())
        indices = torch.argmin(distances, dim=1).unsqueeze(1) # [N, 1]

        # Quantize
        x_quantized = self.embedding(indices).view(x.size())

        # VQ Loss
        # Compute the VQ Losses
        commitment_loss = nn.functional.mse_loss(x_quantized.detach(), x)
        embedding_loss = nn.functional.mse_loss(x_quantized, x.detach())
        vq_loss = commitment_loss * 0.25 + embedding_loss

        x_quantized = x + (x_quantized - x).detach()

        return x_quantized, vq_loss

    
class VAE(nn.Module):
    def __init__(self, latent, num_embeddings):
        super(VAE, self).__init__()
        self.latent = latent
        self.encoder = Encoder(latent)
        self.decoder = Decoder(latent)
        self.vq = VectorQuantization(latent, num_embeddings)
            
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.vq(z)

        return self.decoder(z_q), vq_loss
    
    def generate(self, num_sample):
        model_device = next(self.parameters()).device
        idxes = torch.tensor(list(range(num_sample))).to(model_device)
        z_q = self.vq.embedding(idxes)
        return self.decoder(z_q)

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

def train(net, dataloader, device):
    optimizer = torch.optim.AdamW(net.parameters())
    net.train()
    with tqdm.tqdm(dataloader, ncols=64) as pbar:
        for x, _ in pbar:
            x = x.to(device)
            x_actual, vq_loss = net(x)
            loss = nn.functional.mse_loss(x_actual, x, reduction='mean') + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss {loss.cpu().item():.4f}")

##################################################################################################################################

from matplotlib import pyplot as plt

def predict(net):
    net.eval() # disable drop-out and batch-normalization
    with torch.no_grad():
        x = net.generate(32)

    images = x.cpu()
    fig, axes = plt.subplots(4, 8, figsize=(4, 4))
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
      python vae.py --train --predict
    """

    device = get_device()
    if FLAGS.train:
        print('Train')
        dataloader = get_dataloader()
        net = VAE(latent=2, num_embeddings=32).to(device)
        for i in range(FLAGS.epochs):
            train(net, dataloader, device)
        torch.save(net.state_dict(), 'vqvae.pth')

    if FLAGS.predict:
        print('Predict')
        net = VAE(latent=2, num_embeddings=32).to(device)
        net.load_state_dict(torch.load('vqvae.pth'))
        predict(net)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 3, "Epochs to train")

    app.run(main)