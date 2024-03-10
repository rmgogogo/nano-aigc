# Referenced the following codes and did little modification
#     https://github.com/hkproj/pytorch-llama/blob/main/model.py
#     https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
# Vanilla Transformer is here.
#     https://github.com/rmgogogo/nano-transformers

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional
import math

"""
Model args
"""
@dataclass
class ModelArgs:
    # input
    max_batch_size: int = 100
    max_seq_len: int = 5
    dim: int = 64
    # net
    n_layers: int = 8
    # attention
    n_heads: int = 8
    # RMSNorm
    norm_eps: float = 1e-5
    # device
    device: str = None
    # vocab
    vocab_size: int = 22

"""
Root Mean Square Layer Normalization
https://arxiv.org/abs/1910.07467
"""
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self.norm(x.float()).type_as(x)

class VannilaPE(nn.Module):
    def __init__(self, dim, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

"""
Llama FeedForward
"""
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = int(8 * args.dim / 3)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        # hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.drop = nn.Dropout(p=0.1)
        self.gate = nn.Linear(args.dim, hidden_dim)
        self.up = nn.Linear(args.dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, args.dim)

    def forward(self, x: torch.Tensor):
        x = self.drop(x)
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.gate(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.up(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.down(x)
        return x

"""
One Block of the Transformer Decoder part (GPT block, Llama block)
"""
class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.pe = VannilaPE(args.dim, args.max_seq_len)
        self.attention = nn.MultiheadAttention(embed_dim=args.dim, num_heads=args.n_heads)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor):
        xnorm = self.attention_norm(x)
        xqk = self.pe(xnorm)
        xa, _ = self.attention(xqk, xqk, xnorm)
        h = x + xa
        out = h + self.feed_forward(self.feed_forward_norm(h))
        return out

"""
Transformer Decoder Part (GPT, Llama, Gemma, Mistral etc., here Llama, all of them are similar.) 
"""
class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DecoderBlock(args))

        self.head_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.head_output = nn.Linear(args.dim, self.vocab_size)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, tokens: torch.Tensor):
        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)
        self.drop(h)
        for layer in self.layers:
            h = layer(h)
        h = self.head_norm(h)
        output = self.head_output(h).float()
        return output
    
##################################################################################################################################
import toy
import tqdm

def get_dataloader(batch_size, n_epochs):
    dataset = toy.ToyDataset(transform=toy.TokenizerTransform(), n_epochs=n_epochs)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def get_device():
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def train(n_epochs, model_path='toy_llama.pth', batch_size=100):
    dataloader = get_dataloader(batch_size, n_epochs)
    device = get_device()
    net = Decoder(ModelArgs(max_batch_size=batch_size, max_seq_len=5, dim=32, n_layers=4, n_heads=8, device=device, vocab_size=22))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters())
    net.train()
    with tqdm.tqdm(dataloader, colour="#e034f2") as pbar:
        for o in pbar:
            x = o[:,:-1].to(device)
            t = o[:,1:].to(device)
            y = net(x)
            loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss {loss.cpu().item():.4f}")
    torch.save(net.state_dict(), model_path)

##################################################################################################################################
import toy

def predict(model_path='toy_llama.pth', user_input='1 + 1 ='):
    device = get_device()
    net = Decoder(ModelArgs(max_batch_size=1, max_seq_len=5, dim=32, n_layers=4, n_heads=8, device=device, vocab_size=22))
    net.load_state_dict(torch.load(model_path))
    tokenizer = toy.ToyTokenizer()

    net.eval()
    with torch.no_grad():
        text = user_input
        tokens = tokenizer.tokenize(text)
        x = torch.tensor(tokens, dtype=torch.int)
        y = net(x).argmax(dim=1)
        print('DEBUG: ', tokenizer.detokenize(y))
        char = tokenizer.token2char(y[-1])
        print(text, char)

##################################################################################################################################

from absl import flags
from absl import app

def main(unused_args):
    """
    Samples:
      python llama.py --train --epochs 100 --predict --input "1 + 1 ="
    """
    if FLAGS.train:
        train(n_epochs=FLAGS.epochs)

    if FLAGS.predict:
        predict(user_input=FLAGS.input)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 2000, "Epochs to train")
    flags.DEFINE_string("input", "1 + 1 =", "Input for prediction")

    app.run(main)