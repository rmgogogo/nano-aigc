"""
The advanced version of CLIP. In clip.py, it uses simple text encoding which actually has no text but still OK to try CLIP.

This sample use real text and HuggingFace BERT pretrained text encoder to try CLIP.

This file reuse the codes and trained model from vae.py
"""

from vae import VAE

##################################################################################################################################
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class HfBert():
    def __init__(self, model_name='bert-base-uncased', device='mps:0'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.device = device

    def eval(self, sentences):
        tokens = self.tokenizer(sentences, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
        for key in tokens:
            tokens[key] = tokens[key].to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens).last_hidden_state
            embeddings = outputs.mean(dim=1)
        return embeddings
    
class TextEncoder(nn.Module):
    def __init__(self, latent=2):
        super(TextEncoder, self).__init__()
        self.text_encoder = HfBert()
        self.dense1 = torch.nn.Linear(768, 128)
        self.dense2 = torch.nn.Linear(128, 32)
        self.dense3 = torch.nn.Linear(32, latent)
    def forward(self, x):
        x = self.text_encoder.eval(x)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
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

class TextMnist(torch.utils.data.Dataset):
    def __init__(self):
        self.images, self.labels = self.__load_mnist__()
        self.texts = self.__number2text__(self.labels)
        
    def __load_mnist__(self):
        tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=tf,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100000, shuffle=False, num_workers=0)
        for images, labels in dataloader:
            break
        return images, labels
    
    def __number2text__(self, numbers):
        number_texts = [
            'zero',
            'one',
            'two',
            'three',
            'four',
            'five',
            'six',
            'seven',
            'eight',
            'nine'
        ]
        texts = []
        for number in numbers:
            texts.append(number_texts[number])
        return texts
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.images[index], self.texts[index]
    
def get_dataloader(batch_size):
    dataset = TextMnist()
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def train(n_epochs, batch_size=128, model_path='clip_pro.pth', vae_model_path='vae.pth'):
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
                for images, texts in batch_pbar:
                    images = images.to(device)
                    with torch.no_grad():
                        mean, logvar = vae.encoder(images)
                        images_codes = vae.sampling(mean, logvar)
                    texts_codes = net(texts)

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

import numpy as np
from matplotlib import pyplot as plt

def get_random_texts(num_samples):
    number_texts = [
        'zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine'
    ]
    numbers = np.random.randint(low=0, high=10, size=(num_samples,))
    texts = []
    for i in numbers:
        number_text = number_texts[i]
        texts.append(number_text)
    return texts

def predict(model_path='clip_pro.pth', vae_model_path='vae.pth'):
    device = get_device()
    net = TextEncoder().to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    vae = VAE(latent=2).to(device)
    vae.load_state_dict(torch.load(vae_model_path))
    vae.eval()

    n_samples = 16
    with torch.no_grad():
        texts = get_random_texts(n_samples)
        texts_codes = net(texts)
        images = vae.decoder(texts_codes)

    images = images.cpu()
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(0).numpy(), cmap='gray')
        ax.set_title(f'{texts[i]}')
        ax.axis("off")
    plt.tight_layout()
    plt.show()

##################################################################################################################################

from absl import flags
from absl import app

def main(unused_args):
    """
    Samples:
      python clip_pro.py --train --epochs 10 --predict
    """
    if FLAGS.train:
        train(FLAGS.epochs)

    if FLAGS.predict:
        predict()

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 3, "Epochs to train")

    app.run(main)