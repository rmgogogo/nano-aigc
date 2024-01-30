"""
GAN.
Everything in one file.
"""

##################################################################################################################################
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_chan=100, hidden=64, out_chan=1):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_chan, hidden * 8, 4, 1, 0), # 4x4
            nn.BatchNorm2d(hidden * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden * 8, hidden * 4, 4, 2, 1), # 8x8
            nn.BatchNorm2d(hidden * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, 2, 1), # 16x16
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden * 2, hidden, 4, 2, 1), # 32x32
            nn.BatchNorm2d(hidden),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden, out_chan, kernel_size=1, stride=1, padding=2), # 28x28
            nn.Tanh() # [-1, 1]
        )

    def forward(self, x):
        output = self.decoder(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, in_chan=1, hidden=64):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_chan, hidden, 4, 2, 1), # H/2, W/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden * 2, 4, 2, 1), # H/4, W/4
            nn.BatchNorm2d(hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden * 2, hidden * 4, 4, 2, 1), # H/8, W/8
            nn.BatchNorm2d(hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden * 4, 1, 4, 2, 1), # H/16, W/16
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.classifier(x)
        return output.view(-1, 1).squeeze(1)

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

def train(n_epochs, batch_size=128, latent=8, hidden=16):
    device = get_device()
    dataloader = get_dataloader(batch_size=batch_size)
    
    netG = Generator(latent, hidden, 1).to(device)
    netD = Discriminator(1, hidden).to(device)
    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters())
    optimizerG = torch.optim.Adam(netG.parameters())
    netG.train()
    netD.train()
    with tqdm(range(n_epochs), colour="#00ee00") as epoch_pbar:
        for _ in epoch_pbar:
            with tqdm(dataloader, leave=False, colour="#005500") as batch_pbar:
                for images, _ in batch_pbar:
                    ############################
                    batch_size = images.size(0)
                    positive_labels = torch.full((batch_size,), 1.0, device=device)
                    negative_labels = torch.full((batch_size,), 0.0, device=device)
                    real_image = images.to(device)

                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    netD.zero_grad()
                    # train with real for D
                    output_D = netD(real_image)
                    loss_real = criterion(output_D, positive_labels)
                    loss_real.backward()
                    # train with fake for G
                    noise_latent = torch.randn(batch_size, latent, 1, 1, device=device)
                    fake_images = netG(noise_latent)
                    output_D = netD(fake_images.detach())
                    loss_fake = criterion(output_D, negative_labels)
                    loss_fake.backward()
                    # update weights for D
                    optimizerD.step()
                    
                    # (2) Update G network: maximize log(D(G(z)))
                    netG.zero_grad()
                    output_D = netD(fake_images) # attach for training G
                    loss_g = criterion(output_D, positive_labels) # train G to let it looks like real
                    loss_g.backward()
                    optimizerG.step()
                    batch_pbar.set_description(f'{loss_real.item():.3f}, {loss_fake.item():.3f}, {loss_g.item():.3f}')
            epoch_pbar.set_description(f'{loss_real.item():.3f}, {loss_fake.item():.3f}, {loss_g.item():.3f}')
    torch.save(netG.state_dict(), 'gan_g.pth')
    torch.save(netD.state_dict(), 'gan_d.pth')

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

def predict(n_samples=64, latent=8, hidden=16):
    device = get_device()
    net = Generator(latent, hidden, 1).to(device)
    net.load_state_dict(torch.load('gan_g.pth'))

    net.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent, 1, 1).to(device)
        x = net(z)
    show_images(x)


##################################################################################################################################

from absl import flags
from absl import app

def main(unused_args):
    """
    Samples:
      python gan.py --train --epochs 100 --predict
    """
    if FLAGS.train:
        train(n_epochs=FLAGS.epochs)

    if FLAGS.predict:
        predict()

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 3, "Epochs to train")
    
    app.run(main)