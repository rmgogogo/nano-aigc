'''
Gemma. Compared with Llama, the only difference is that in gated FeedForward, Gemma uses GELU while Llama uses SILU.
No performance difference. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class RMSNorm(nn.Module):
    '''
    Root Mean Square Layer Normalization, https://arxiv.org/abs/1910.07467
    Trick: RMSNorm is 15.9/14.17 = 1.12X faster than GPT2 LayerNorm
    '''
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
class RoPE(nn.Module):
    '''
    Rotary Position Encoding, https://arxiv.org/abs/2104.09864
    Trick: RoPE performance is much better than other position encoding.
    Trick: RoPE once or RoPE in each attention's Q and K has no difference.
    '''
    def __init__(self, max_seq, embed_dim, theta = 10000.0):
        super().__init__()
        self.register_buffer("freqs_complex", self.pre_calc(max_seq, embed_dim, theta))

    def pre_calc(self, max_seq, embed_dim, theta):
        theta_numerator = torch.arange(0, embed_dim, 2)
        theta = 1.0 / (theta ** (theta_numerator / embed_dim))
        position = torch.arange(max_seq)
        freqs = torch.outer(position, theta)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        freqs_complex = freqs_complex.unsqueeze(0)
        return freqs_complex

    def forward(self, x):
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * self.freqs_complex
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)
        return x_out

class GemmaBlock(nn.Module):
    '''
    Gemma Block
    '''
    def __init__(self, embed_dim, num_heads, dropout, max_seq):
        super().__init__()
        self.ln1 = RMSNorm(embed_dim)
        # Trick: when using nn.MultiheadAttention, take care the batch_first and attn_mask
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.register_buffer("attention_mask", torch.tril(torch.ones((max_seq, max_seq))) == 0)
        self.ln2 = RMSNorm(embed_dim)
        self.ff_gate = nn.Linear(embed_dim, 2*embed_dim)
        self.ff_in_proj = nn.Linear(embed_dim, 2*embed_dim)
        self.ff_out_proj = nn.Linear(2*embed_dim, embed_dim)
        self.ff_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Trick: put FF before attention has no difference. Maybe it's too expensive for people to try.
        res = self.ln1(x)
        res, _ = self.attention(res, res, res, attn_mask=self.attention_mask)
        x = x + res
        
        # Trick: Gemma/Llama feed-forward is 14.17/12.8=1.10x faster than GPT2 feed-forward.
        res = self.ln2(x)
        gate = F.gelu(self.ff_gate(res))
        v = self.ff_in_proj(res)
        res = gate * v
        res = self.ff_out_proj(res)
        res = self.ff_dropout(res)
        x = x + res
        return x

class Gemma(nn.Module):
    '''
    LLaMA
    '''
    def __init__(self, n_blocks, n_vocab, max_seq, embed_dim, num_heads, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, embed_dim)
        self.positioning = RoPE(max_seq=max_seq, embed_dim=embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(GemmaBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, max_seq=max_seq))
        self.final_ln = RMSNorm(embed_dim)
        self.final_dense = nn.Linear(embed_dim, n_vocab)

    def forward(self, tokens):
        # [B, S]
        x = self.token_embedding(tokens)
        # [B, S, C]
        x = self.positioning(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        # [B, S, C]
        x = self.final_dense(x)
        # [B, S, V]
        return x

##################################################################################################################################
import toy
import tqdm

def get_dataloader(batch_size, max_seq, n_epochs):
    dataset = toy.ToyDataset(transform=toy.TokenizerTransform(max_seq=max_seq), n_epochs=n_epochs)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

def get_device():
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def train(n_epochs, batch_size=100, max_seq=5, embed_dim=64, n_vocab=22, n_blocks=8, num_heads=8, dropout=0.1, model_path='llama.pth', comment=''):
    dataloader = get_dataloader(batch_size, max_seq+1, n_epochs)
    device = get_device()
    net = Gemma(n_blocks=n_blocks, n_vocab=n_vocab, max_seq=max_seq, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    writer = SummaryWriter(comment=comment)
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        x = batch[:,:-1].to(device)
        t = batch[:,1:].to(device)
        y = net(x)
        # CPU requires contiguous(), but MPS and CUDA OK.
        loss = F.cross_entropy(y.contiguous().view(-1, y.shape[-1]), t.contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Accuracy
        truth = t[:,3]
        actual = torch.argmax(y, dim=2)[:,3]
        accuracy = (actual == truth).sum().item() / truth.shape[0]
        # TensorBoard
        writer.add_scalar("Accuracy", accuracy, batch_idx)
        writer.add_scalar("Loss", loss.item(), batch_idx)
        # Can't work in CUDA but OK in MPS
        # if batch_idx == 0:
        #     writer.add_graph(net, input_to_model=x, verbose=False)
        if batch_idx == n_epochs-1:
            for pn, p in net.named_parameters():
                writer.add_histogram(pn, p, global_step=batch_idx)
    torch.save(net.state_dict(), model_path)

##################################################################################################################################

def predict(user_input='1 + 1 =', max_seq=5, embed_dim=64, n_vocab=22, n_blocks=8, num_heads=8, dropout=0.1, model_path='llama.pth'):
    device = get_device()
    net = Gemma(n_blocks=n_blocks, n_vocab=n_vocab, max_seq=max_seq, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    tokenizer = toy.ToyTokenizer()
    tokenizer_transform = toy.TokenizerTransform(max_seq=max_seq)
    net.eval()
    with torch.no_grad():
        text = user_input
        x = tokenizer_transform(text)
        x = x.unsqueeze(0).to(device)
        y = net(x)
        y = y.argmax(dim=2)[0].cpu()
        char = tokenizer.token2char(y[3])
        print(text, char)

##################################################################################################################################

from absl import flags
from absl import app

def main(unused_args):
    """
    Samples:
      python gemma.py --train --epochs 200 --comment "train-comment" --predict --input "1 + 1 ="
    """
    if FLAGS.train:
        # Trick: more block performance always is better, more attention head doesn't help a lot
        for head in [8]: # [2, 4, 8, 16]:
            for block in [8]: # [2, 4, 8, 16]:
                comment = f'-h-{head}-b-{block}'
                train(n_epochs=FLAGS.epochs, comment=comment, num_heads=head, n_blocks=block)

    if FLAGS.predict:
        predict(user_input=FLAGS.input, num_heads=head, n_blocks=block)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 200, "Epochs to train")
    flags.DEFINE_string("input", "1 + 1 =", "Input for prediction")

    app.run(main)