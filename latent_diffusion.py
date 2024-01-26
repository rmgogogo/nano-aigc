"""
Latent Diffusion.
- Use VAE in vae.py for image encoding and decoding.
- Use Diffusion in Latent space, together with the CLIP condition to generate a latent. Since it's nano, no cross attention here.
- Use simple embedding to simulate the CLIP condition.
"""

##################################################################################################################################
import torch
import torch.nn as nn

class Net(nn.Module):
    """
      Predict the noise from a noised latent tensor [N, latent].
      Backbone is not UNet, but a simple MLP.
    """
    def __init__(self, n_steps=1000, latent=8):
        super(Net, self).__init__()
        self.time_embed = nn.Embedding(n_steps, latent)
        self.condition_embed = nn.Embedding(10, latent)
        self.l1 = nn.Linear(latent, 4)
        self.l2 = nn.Linear(4, 2)
        self.l3 = nn.Linear(2, 4)
        self.l4 = nn.Linear(8, latent)
        self.l5 = nn.Linear(latent, latent)
        self.te1 = nn.Linear(latent*2, latent)
        self.te2 = nn.Linear(latent*2, 4)
        self.te3 = nn.Linear(latent*2, 2)
        self.te4 = nn.Linear(latent*2, 8)

    def forward(self, x, t, c):
        t = self.time_embed(t).squeeze(1)
        c = self.condition_embed(c).squeeze(1)
        t = torch.cat((t, c), dim=1)

        x = self.l1(x + self.te1(t)) #[N, 4]
        x1 = x = torch.relu(x)
        x = self.l2(x + self.te2(t)) #[N, 2]
        x = torch.relu(x)
        x = self.l3(x + self.te3(t)) #[N, 4]
        x = torch.relu(x)
        x = torch.cat((x, x1), dim=1) #[N,8]
        x = self.l4(x + self.te4(t)) #[N, 8]
        x = torch.relu(x)
        x = self.l5(x)
        return x

##################################################################################################################################
from tqdm.auto import tqdm
from vae import VAE
from diffusion import Noiser
import torchvision

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

def train(n_epochs, batch_size=128, n_steps=1000, beta_min=0.0001, beta_max=0.02, latent=8, model_path='latent_diffusion.pth', ae_model_path='vae_8.pth'):
    device = get_device()
    dataloader = get_dataloader(batch_size=batch_size)
    noiser = Noiser(device=device, n_steps=n_steps, beta_min=beta_min, beta_max=beta_max)
    net = Net(n_steps=n_steps, latent=latent).to(device)
    optim = torch.optim.Adam(net.parameters())

    ae = VAE(latent=latent).to(device)
    ae.load_state_dict(torch.load(ae_model_path))
    ae.eval()
    
    net.train()
    with tqdm(range(n_epochs), colour="#00ee00") as epoch_pbar:
        for _ in epoch_pbar:
            with tqdm(dataloader, leave=False, colour="#005500") as batch_pbar:
                for images, labels in batch_pbar:
                    images = images.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        x0 = ae.calc_latent(images)
                    cur_batch_size = len(images)
                    t = torch.randint(0, n_steps, (cur_batch_size,)).to(device)
                    xt, epsilon = noiser.noisy_1d(x0, t)
                    epsilon_hat = net(xt, t.reshape(cur_batch_size, -1), labels)
                    loss = nn.functional.mse_loss(epsilon_hat, epsilon)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    batch_pbar.set_description(f'{loss.item():.3f}')
            epoch_pbar.set_description(f'{loss.item():.3f}')
    torch.save(net.state_dict(), model_path)

##################################################################################################################################
from matplotlib import pyplot as plt

def predict(ddim_steps=20, eta=1, n_steps=1000, beta_min=0.0001, beta_max=0.02, latent=8, model_path='latent_diffusion.pth', ae_model_path='vae_8.pth'):
    device = get_device()

    ae = VAE(latent=latent).to(device)
    ae.load_state_dict(torch.load(ae_model_path))
    ae.eval()

    noiser = Noiser(device=device, n_steps=n_steps, beta_min=beta_min, beta_max=beta_max)
    net = Net(n_steps=n_steps, latent=latent).to(device)

    n_samples = 16
    with torch.no_grad():
        labels = torch.randint(low=0, high=10, size=(n_samples,)).to(device)
        
        # diffusion the x
        x = torch.rand(n_samples, latent).to(device)
        ts = torch.linspace(n_steps, 0, (ddim_steps + 1)).to(torch.long).to(device)
        for i in tqdm(range(ddim_steps)):
            cur_t = ts[i] - 1 # 999
            prev_t = ts[i+1] - 1 # 949
            time_tensor = (torch.ones(n_samples, 1).to(device) * cur_t).long()
            epislon = net(x, time_tensor, labels)
            noise = torch.randn_like(x)

            ab_cur = noiser.alpha_bars[cur_t]
            ab_prev = noiser.alpha_bars[prev_t] if prev_t >= 0 else 1
            var = eta * torch.sqrt((1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev))
            w1 = (ab_prev / ab_cur)**0.5
            w2 = (1 - ab_prev - var**2)**0.5 - (ab_prev * (1 - ab_cur) / ab_cur)**0.5
            w3 = var
            x = w1 * x + w2 * epislon + w3 * noise
        images = ae.decoder(x)

    images = images.cpu()
    _, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(0).numpy(), cmap='gray')
        ax.set_title(f'{labels[i]}')
        ax.axis("off")
    plt.tight_layout()
    plt.show()

##################################################################################################################################
from absl import flags
from absl import app

def main(unused_args):
    """
    Samples:
      python latent_diffusion.py --train --epochs 100 --predict
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