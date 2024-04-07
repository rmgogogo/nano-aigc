'''
Vannila Transformer Decoder, aka GPT2
'''
# TODO: accuracy, use pre-built attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter

class Positioning(nn.Module):
    '''
    Learnable Position Embedding
    '''
    def __init__(self, embed_dim, max_seq, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(1, max_seq, embed_dim))

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class CausalSelfAttention(nn.Module):
    """
    Copied from miniGPT
    """

    def __init__(self, n_embd, n_head, dropout, max_seq):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seq, max_seq)).view(1, 1, max_seq, max_seq))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class GPT2Block(nn.Module):
    '''
    GPT2 Blocker
    '''
    def __init__(self, embed_dim, num_heads, dropout, max_seq):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.attention = CausalSelfAttention(n_embd=embed_dim, n_head=num_heads, dropout=dropout, max_seq=max_seq)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff_in_proj = nn.Linear(embed_dim, 2*embed_dim)
        self.ff_out_proj = nn.Linear(2*embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        res = self.ln1(x)
        #res = self.attention(res, res, res, need_weights=False)[0]
        res = self.attention(res)
        x = x + res
        res = self.ln2(x)
        res = self.ff_in_proj(res)
        res = F.gelu(res)
        res = self.ff_out_proj(res)
        res = self.dropout(res)
        x = x + res
        return x

class GPT2(nn.Module):
    '''
    GPT2
    '''
    def __init__(self, n_blocks, n_vocab, max_seq, embed_dim, num_heads, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, embed_dim)
        self.positioning = Positioning(embed_dim=embed_dim, max_seq=max_seq, dropout=dropout)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(GPT2Block(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, max_seq=max_seq))
        self.final_ln = nn.LayerNorm(embed_dim)
        self.final_dense = nn.Linear(embed_dim, n_vocab)

        # Trick: initialize the weights
        self.apply(self._init_weights)
        # Tick: initialize the weights based on residential block numbers
        for pn, p in self.named_parameters():
            if pn.endswith('ff_out_proj.weight') or pn.endswith('out_proj.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_blocks))
            if pn.endswith('positioning.pe'):
                torch.nn.init.uniform_(p)

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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

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

def train(n_epochs, batch_size=100, max_seq=5, embed_dim=64, n_vocab=22, n_blocks=8, num_heads=8, dropout=0.1, model_path='gpt2.pth'):
    dataloader = get_dataloader(batch_size, max_seq+1, n_epochs)
    device = get_device()
    net = GPT2(n_blocks=n_blocks, n_vocab=n_vocab, max_seq=max_seq, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    writer = SummaryWriter()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        x = batch[:,:-1].to(device)
        t = batch[:,1:].to(device)
        y = net(x)
        loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # TensorBoard
        writer.add_scalar("Loss", loss.cpu().item(), batch_idx)
        if batch_idx == 0:
            writer.add_graph(net, input_to_model=x, verbose=False)
        if batch_idx == n_epochs-1:
            for pn, p in net.named_parameters():
                writer.add_histogram(pn, p, global_step=batch_idx)
    torch.save(net.state_dict(), model_path)

##################################################################################################################################

def predict(user_input='1 + 1 =', max_seq=5, embed_dim=64, n_vocab=22, n_blocks=8, num_heads=8, dropout=0.1, model_path='gpt2.pth'):
    device = get_device()
    net = GPT2(n_blocks=n_blocks, n_vocab=n_vocab, max_seq=max_seq, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
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
      python gpt2.py --train --epochs 10000 --predict --input "1 + 1 ="
    """
    if FLAGS.train:
        train(n_epochs=FLAGS.epochs)

    if FLAGS.predict:
        predict(user_input=FLAGS.input)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Predict")
    flags.DEFINE_integer("epochs", 10, "Epochs to train")
    flags.DEFINE_string("input", "1 + 1 =", "Input for prediction")

    app.run(main)