# Referenced the following codes and did little modification
#     https://github.com/hkproj/pytorch-llama/blob/main/model.py
#     https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
# Vanilla Transformer is here.
#     https://github.com/rmgogogo/nano-transformers
# TODO: nano-training

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
    max_seq_len: int = 6
    dim: int = 32
    # net
    n_layers: int = 4
    # Attention
    #   query head
    n_heads: int = 8
    #   key value head, default to query head
    n_kv_heads: Optional[int] = None
    # FeedForward
    multiple_of: int = 256
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

"""
Rotary Position Embedding
https://arxiv.org/abs/2104.09864
"""
class RoPE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # TODO: consider better way to wrap the RoPE
        self.freqs_complex = nn.Parameter(
            self.precompute_theta_pos_frequencies(args.dim // args.n_heads, args.max_seq_len * 2, args.device),
            requires_grad=False)

    def precompute_theta_pos_frequencies(self, dim_per_head: int, seq_len: int, device: str, theta: float = 10000.0):
        # Build the theta parameter
        # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
        # Shape: (dim_per_head / 2)
        theta_numerator = torch.arange(0, dim_per_head, 2).float()
        # Shape: (dim_per_head / 2)
        theta = 1.0 / (theta ** (theta_numerator / dim_per_head)).to(device) # (Dim / 2)
        # Construct the positions (the "m" parameter)
        # Shape: (Seq_Len)
        m = torch.arange(seq_len, device=device)
        # Multiply each theta by each position using the outer product.
        # Shape: (Seq_Len) outer_product* (dim_per_head / 2) -> (Seq_Len, dim_per_head / 2)
        freqs = torch.outer(m, theta).float()
        # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
        # (Seq_Len, dim_per_head / 2) -> (Seq_Len, dim_per_head / 2)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. 
        # So we need to add the batch dimension and the head dimension
        # (Seq_Len, dim_per_head/2) --> (1, Seq_Len, 1, dim_per_head/2)
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
        return freqs_complex

    def forward(self, x: torch.Tensor, start_pos: int):
        with torch.no_grad():
            seq_len = x.shape[1]
            fq_complex = self.freqs_complex[:,start_pos:start_pos+seq_len,:,:]
            # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
            # Two consecutive values will become a single complex number
            # (B, Seq_Len, H, dim_per_head) -> (B, Seq_Len, H, dim_per_head/2)
            x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
            # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
            # (B, Seq_Len, H, dim_per_head/2) * (1, Seq_Len, 1, dim_per_head/2) = (B, Seq_Len, H, dim_per_head/2)
            x_rotated = x_complex * fq_complex
            # Convert the complex number back to the real number
            # (B, Seq_Len, H, dim_per_head/2) -> (B, Seq_Len, H, dim_per_head/2, 2)
            x_out = torch.view_as_real(x_rotated)
            # (B, Seq_Len, H, dim_per_head/2, 2) -> (B, Seq_Len, H, dim_per_head)
            x_out = x_out.reshape(*x.shape)
            return x_out.type_as(x).to(x.device)

"""
Multi Groupled Multi Query Muti Head Self Attention + KV Cache
"""
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.dim_per_head = args.dim // args.n_heads

        # TODO: share rope cross blocks
        self.rope = RoPE(args)
        self.wq = nn.Linear(args.dim, args.n_heads * self.dim_per_head, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.dim_per_head, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.dim_per_head, bias=False)
        self.wo = nn.Linear(args.n_heads * self.dim_per_head, args.dim, bias=False)

        # TODO: in training mode, disable KV Cache? check whether it's the root cause
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.dim_per_head)).to(args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.dim_per_head)).to(args.device)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, seq_len, n_kv_heads, dim_per_head = x.shape
        if n_rep == 1:
            return x
        return (
            # (B, Seq_Len, N_KV_Heads, 1, dim_per_head)
            x[:, :, :, None, :]
            # (B, Seq_Len, N_KV_Heads, N_Rep, dim_per_head)
            .expand(batch_size, seq_len, n_kv_heads, n_rep, dim_per_head)
            # (B, Seq_Len, N_KV_Heads * N_Rep, dim_per_head)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, dim_per_head)
        )

    def forward(self, x: torch.Tensor, start_pos: int):
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * dim_per_head)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * dim_per_head)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * dim_per_head)
        xv = self.wv(x)

        # (B, 1, H_Q * dim_per_head) -> (B, 1, H_Q, dim_per_head)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.dim_per_head)
        # (B, 1, H_KV * dim_per_head) -> (B, 1, H_KV, dim_per_head)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.dim_per_head)
        # (B, 1, H_KV * dim_per_head) -> (B, 1, H_KV, dim_per_head)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.dim_per_head)

        # (B, 1, H_Q, dim_per_head) --> (B, 1, H_Q, dim_per_head)
        xq = self.rope(xq, start_pos)
        # (B, 1, H_KV, dim_per_head) --> (B, 1, H_KV, dim_per_head)
        xk = self.rope(xk, start_pos)

        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
        # (B, Seq_Len_KV, H_KV, dim_per_head)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, dim_per_head)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, dim_per_head) --> (B, Seq_Len_KV, H_Q, dim_per_head)
        keys = self.repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, dim_per_head) --> (B, Seq_Len_KV, H_Q, dim_per_head)
        values = self.repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, dim_per_head) -> (B, H_Q, 1, dim_per_head)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, dim_per_head) -> (B, H_Q, Seq_Len_KV, dim_per_head)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, dim_per_head) -> (B, H_Q, Seq_Len_KV, dim_per_head)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, dim_per_head) @ (B, H_Q, dim_per_head, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.dim_per_head)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, dim_per_head) -> (B, H_Q, 1, dim_per_head)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, dim_per_head) -> (B, 1, H_Q, dim_per_head) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

"""
Llama FeedForward
"""
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = int(8 * args.dim / 3)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.gate = nn.Linear(args.dim, hidden_dim, bias=False)
        self.up = nn.Linear(args.dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor):
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

        self.n_heads = args.n_heads
        self.dim = args.dim

        self.self_attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.self_attention(self.attention_norm(x), start_pos)
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
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
        self.head_output = nn.Linear(args.dim, self.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, start_pos)
        h = self.head_norm(h)
        output = self.head_output(h).float()
        return output
    
##################################################################################################################################
import toy
import tqdm

def get_dataloader(batch_size):
    dataset = toy.ToyDataset(transform=toy.TokenizerTransform())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def get_device():
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def train(n_epochs, model_path='toy_llama.pth', batch_size=16):
    dataloader = get_dataloader(batch_size)
    device = get_device()
    net = Decoder(ModelArgs(max_batch_size=batch_size, max_seq_len=6, dim=32, n_layers=4, n_heads=8, device=device, vocab_size=22))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters())
    net.train()
    for _ in range(n_epochs):
        with tqdm.tqdm(dataloader, colour="#e034f2") as pbar:
            for o in pbar:
                x = o[:,:-1].to(device)
                t = o[:,1:].to(device)
                y = net(x, 0)
                loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Loss {loss.cpu().item():.4f}")
    torch.save(net.state_dict(), model_path)


##################################################################################################################################

from absl import flags
from absl import app

def main(unused_args):
    """
    Samples:
      python llama.py --train --epochs 100 --predict
    """
    if FLAGS.train:
        train(n_epochs=FLAGS.epochs)

    # if FLAGS.predict:
    #     predict()

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 3, "Epochs to train")

    app.run(main)