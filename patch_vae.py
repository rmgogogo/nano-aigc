"""
Train VAE based on 4x4 Image Pitches and VQ. It's the preparation for ViT, DiT, Sora etc.
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
    def __init__(self, latent=2):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            Flatten(),
            nn.Linear(4*4, 6),
            nn.ReLU(),
            nn.Linear(6, latent),
            nn.Tanh()
        )
    def forward(self, x):
        return self.encode(x)

class Decoder(nn.Module):
    def __init__(self, latent=2):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(in_features=latent, out_features=6),
            nn.ReLU(),
            nn.Linear(in_features=6, out_features=4*4),
            nn.Tanh(),
            Reshape((-1, 1, 4, 4))
        )
    def forward(self, x):
        return self.decode(x)

# Vector Quantization Layer
class VectorQuantization(nn.Module):
    def __init__(self, embedding_dim, num_codebook):
        super(VectorQuantization, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_codebook, embedding_dim)
        self.embedding.weight.data.uniform_(-1, 1)

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
    def __init__(self, latent, num_codebook):
        super(VAE, self).__init__()
        self.latent = latent
        self.encoder = Encoder(latent)
        self.decoder = Decoder(latent)
        self.vq = VectorQuantization(latent, num_codebook)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.vq(z)

        return self.decoder(z_q), vq_loss

##################################################################################################################################
import torchvision
import tqdm
import einops

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
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def train(net, dataloader, device):
    optimizer = torch.optim.AdamW(net.parameters())
    net.train()
    with tqdm.tqdm(dataloader, ncols=64) as pbar:
        for x, _ in pbar:
            x = x.to(device)
            x = einops.rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = 4, p2 = 4)
            x_actual, vq_loss = net(x)
            loss = nn.functional.mse_loss(x_actual, x, reduction='mean') + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss {loss.cpu().item():.4f}")

##################################################################################################################################

from matplotlib import pyplot as plt

def unpatch(x, grid_size=(7, 7), batch_size=128):
    _, c, ph, pw = x.size()
    x = x.view(batch_size, -1, c, ph, pw)
    batch_size, num_patches, c, ph, pw = x.size()
    assert num_patches == grid_size[0] * grid_size[1]
    x_image = x.view(batch_size, grid_size[0], grid_size[1], c, ph, pw)
    output_h = grid_size[0] * ph
    output_w = grid_size[1] * pw
    x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_image = x_image.view(batch_size, c, output_h, output_w)
    return x_image

def predict(net, device):
    dataloader = get_dataloader()
    for batch in dataloader:
        images, _ = batch
        break
    images = images.to(device)
    patchs = einops.rearrange(images, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = 4, p2 = 4)

    net.eval()
    with torch.no_grad():
        x_h, _ = net(patchs)
    x_h = x_h.to('cpu')

    xx = unpatch(x_h)
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(xx[i].squeeze(0).numpy(), cmap='gray', vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

##################################################################################################################################

from absl import flags
from absl import app

def main(unused_args):
    """
    Samples:
      python patch_vae.py --train --predict --num_codebook 4096 --latent 4
    """

    device = get_device()
    if FLAGS.train:
        print('Train')
        dataloader = get_dataloader()
        net = VAE(latent=FLAGS.latent, num_codebook=FLAGS.num_codebook).to(device)
        for i in range(FLAGS.epochs):
            train(net, dataloader, device)
        torch.save(net.state_dict(), 'patch_vae.pth')

    if FLAGS.predict:
        print('Predict')
        net = VAE(latent=FLAGS.latent, num_codebook=FLAGS.num_codebook).to(device)
        net.load_state_dict(torch.load('patch_vae.pth'))
        predict(net, device)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 5, "Epochs to train")
    flags.DEFINE_integer("num_codebook", 4096, "Codebook size")
    flags.DEFINE_integer("latent", 4, "Latent dimention")

    app.run(main)