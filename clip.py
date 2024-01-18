"""
CLIP to put text and image into one space.
This program use a trained VAE encoder to encode the image, use a simple embedding to simulate the text encoding, 
use the CLIP to align the text encoding into image space.
"""

##################################################################################################################################
#### This part is copied from vae.py, just to demo all in one file for simplicity
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
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU()
        )
        self.calc_mean = nn.Linear(256, latent)
        self.calc_logvar = nn.Linear(256, latent)
    
    def forward(self, x):
        x = self.encode(x)
        return self.calc_mean(x), self.calc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent=2):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(in_features=latent, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=28*28),
            nn.Sigmoid(),
            Reshape((-1, 1, 28, 28))
        )
    def forward(self, x):
        return self.decode(x)
    
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
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, num_class=10, latent=2):
        super(TextEncoder, self).__init__()
        self.text_encoder = torch.nn.Embedding(num_class, 32)
        self.dense1 = torch.nn.Linear(32, 16)
        self.dense2 = torch.nn.Linear(16, latent)
        # self.logit_scale = torch.nn.Parameter(torch.ones([]))
    def forward(self, x):
        x = self.text_encoder(x)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        
        # Scale doesn't help here.
        # x = self.logit_scale * x
        return x

##################################################################################################################################
import torchvision
from tqdm.auto import tqdm

def get_device():
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps:0'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def get_dataloader(batch_size):
    tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def train(n_epochs, batch_size=128, model_path='clip.pth', vae_model_path='vae.pth'):
    device = get_device()
    dataloader = get_dataloader(batch_size=batch_size)
    net = TextEncoder().to(device)
    optim = torch.optim.Adam(net.parameters())

    vae = VAE(latent=2).to(device)
    vae.load_state_dict(torch.load(vae_model_path))
    vae.eval()

    net.train()
    with tqdm(range(n_epochs), colour="#00ee00") as epoch_pbar:
        for _ in epoch_pbar:
            with tqdm(dataloader, leave=False, colour="#005500") as batch_pbar:
                for images, labels in batch_pbar:
                    images = images.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        mean, logvar = vae.encoder(images)
                        images_codes = vae.sampling(mean, logvar)
                    texts_codes = net(labels)

                    # L2-norm doesn't help here.
                    # images_codes = images_codes / images_codes.norm(dim=-1, keepdim=True)
                    # texts_codes = texts_codes / texts_codes.norm(dim=-1, keepdim=True)

                    logits = images_codes @ texts_codes.t()
                    target_logits = torch.arange(len(images)).to(device)
                    loss1 = torch.nn.functional.cross_entropy(logits, target_logits)
                    loss2 = torch.nn.functional.cross_entropy(logits.t(), target_logits)
                    loss = (loss1 + loss2) / 2

                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    batch_pbar.set_description(f'{loss.item():.3f}')
    torch.save(net.state_dict(), model_path)

##################################################################################################################################

from matplotlib import pyplot as plt

def predict(model_path='clip.pth', vae_model_path='vae.pth'):
    device = get_device()
    net = TextEncoder().to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    vae = VAE(latent=2).to(device)
    vae.load_state_dict(torch.load(vae_model_path))
    vae.eval()

    n_samples = 16
    with torch.no_grad():
        labels = torch.randint(low=0, high=10, size=(n_samples,)).to(device).reshape(n_samples, -1)
        texts_codes = net(labels)
        images = vae.decoder(texts_codes)

    images = images.cpu()
    labels = labels.cpu()
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(0).numpy(), cmap='gray')
        ax.set_title(f'{labels[i][0]}')
        ax.axis("off")
    plt.tight_layout()
    plt.show()

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
      python clip.py --train --epochs 10 --predict
    """
    if FLAGS.train:
        train(FLAGS.epochs)

    if FLAGS.predict:
        predict()

if __name__ == '__main__':
    app.run(main)