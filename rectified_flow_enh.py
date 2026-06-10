"""
Rectified Flow with PyTorch — ViT velocity network (DiT-style).
Everything in one file.

Replaces the UNet VelocityNet with a Vision Transformer (VelocityViT):
  - Images are split into 4×4 patches → 7×7 = 49 tokens for 28×28 MNIST.
  - Time conditioning via AdaLN (Adaptive Layer Norm), same trick as DiT.
  - Transformer blocks with multi-head self-attention + MLP.
  - Tokens are unpatchified back to image space as the velocity output.

Rectified Flow (Liu et al. 2022) builds on Flow Matching with one key addition — Reflow:
  Round 1: Train with independent coupling (identical to Flow Matching)
  Reflow:  Run the trained ODE on noise samples to collect causally-coupled (x0, x1) pairs,
           then retrain on those pairs. Coupled paths don't cross → straighter velocity field
           → fewer ODE steps needed at inference.

Usage:
  python rectified_flow_enh.py --train --epochs 100        # Round 1
  python rectified_flow_enh.py --reflow --epochs 100       # Reflow → Round 2
  python rectified_flow_enh.py --predict                   # Generate (uses r2 if available, else r1)
  python rectified_flow_enh.py --predict --fast            # Fewer ODE steps
"""

##################################################################################################################################
import os
import math
import torch
import torch.nn as nn


class FlowMatcher:
    """Linear interpolation between noise (x0) and data (x1)."""

    def interpolate(self, x1, t):
        # Round 1: x0 sampled independently — same as Flow Matching.
        # x1: (N, C, H, W), t: (N,) in [0, 1]
        n = len(x1)
        x0 = torch.randn_like(x1)
        t_view = t.reshape(n, 1, 1, 1)
        xt = (1 - t_view) * x0 + t_view * x1
        return xt, x1 - x0, x0

    def interpolate_coupled(self, x0, x1, t):
        # Reflow: x0 and x1 are causally paired — paths won't cross.
        # x0, x1: (N, C, H, W), t: (N,) in [0, 1]
        n = len(x1)
        t_view = t.reshape(n, 1, 1, 1)
        xt = (1 - t_view) * x0 + t_view * x1
        return xt, x1 - x0


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t in [0, 1]."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        if t.dim() > 1:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


# ── ViT building blocks ──────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to embed_dim."""
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (N, C, H, W) → (N, n_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class AdaLN(nn.Module):
    """Adaptive Layer Norm: scale and shift are conditioned on time embedding."""
    def __init__(self, embed_dim, time_emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(time_emb_dim, 2 * embed_dim)

    def forward(self, x, t_emb):
        # t_emb: (N, time_emb_dim)
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)  # each (N, embed_dim)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Transformer block with AdaLN time conditioning."""
    def __init__(self, embed_dim, n_heads, time_emb_dim, mlp_ratio=4.0):
        super().__init__()
        self.attn_norm = AdaLN(embed_dim, time_emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ff_norm = AdaLN(embed_dim, time_emb_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x, t_emb):
        normed = self.attn_norm(x, t_emb)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.ff(self.ff_norm(x, t_emb))
        return x


class VelocityViT(nn.Module):
    """
    ViT velocity field predictor (DiT-style).

    Architecture:
      PatchEmbed → learned pos embed → N × TransformerBlock(AdaLN) → head → unpatchify
    """
    def __init__(self, img_size=28, patch_size=4, in_channels=1,
                 embed_dim=256, depth=6, n_heads=8,
                 time_emb_dim=256, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = img_size // patch_size  # 7 for 28×28 / patch 4

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size ** 2, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, time_emb_dim, mlp_ratio)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def unpatchify(self, x):
        """(N, n_patches, p*p*C) → (N, C, H, W)"""
        p, c, g = self.patch_size, self.in_channels, self.grid_size
        x = x.reshape(x.shape[0], g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (N, C, g, p, g, p)
        return x.reshape(x.shape[0], c, g * p, g * p)

    def forward(self, x, t):
        t_emb = self.time_embed(t)                     # (N, time_emb_dim)
        x = self.patch_embed(x) + self.pos_embed       # (N, n_patches, embed_dim)
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.final_norm(x)
        x = self.head(x)                               # (N, n_patches, p*p*C)
        return self.unpatchify(x)                      # (N, C, H, W)


class CoupledDataset(torch.utils.data.Dataset):
    """Holds causally-coupled (x0, x1) pairs produced by reflow."""
    def __init__(self, x0, x1):
        self.x0 = x0  # (N, 1, 28, 28) on CPU
        self.x1 = x1

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx]


##################################################################################################################################
import torchvision
from tqdm.auto import tqdm

R1_PATH = 'rectified_flow_enh_r1.pth'
R2_PATH = 'rectified_flow_enh_r2.pth'


def get_dataloader(batch_size=128):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.mnist.MNIST("./data", download=True, train=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def get_device():
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps:0'
    if torch.cuda.is_available():
        device = 'cuda:0'
    return device


def _make_net(device, time_emb_dim=256):
    return VelocityViT(time_emb_dim=time_emb_dim).to(device)


def _euler(net, x, n_steps, device):
    """Shared Euler ODE integration loop."""
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((len(x),), i * dt, device=device)
        x = x + dt * net(x, t)
    return x


def train(n_epochs, batch_size=128, time_emb_dim=256, model_path=R1_PATH):
    """Round 1: independent coupling, identical to Flow Matching."""
    device = get_device()
    dataloader = get_dataloader(batch_size=batch_size)
    flow_matcher = FlowMatcher()
    net = _make_net(device, time_emb_dim)
    optim = torch.optim.Adam(net.parameters())

    net.train()
    with tqdm(range(n_epochs), colour="#00ee00") as epoch_pbar:
        for _ in epoch_pbar:
            with tqdm(dataloader, leave=False, colour="#005500") as batch_pbar:
                for images, _ in batch_pbar:
                    x1 = images.to(device)
                    n = len(x1)
                    t = torch.rand(n, device=device)
                    xt, velocity, _ = flow_matcher.interpolate(x1, t)
                    loss = nn.functional.mse_loss(net(xt, t), velocity)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    batch_pbar.set_description(f'{loss.item():.3f}')
    torch.save(net.state_dict(), model_path)


def generate_reflow_pairs(n_samples=60000, n_steps=100, batch_size=256,
                          time_emb_dim=256, r1_path=R1_PATH):
    """
    Run the Round-1 ODE on noise to collect causally-coupled (x0, x1) pairs.
    Each x0 deterministically maps to one x1, so their connecting paths never cross.
    """
    device = get_device()
    net = _make_net(device, time_emb_dim)
    net.load_state_dict(torch.load(r1_path, map_location=device))
    net.eval()

    all_x0, all_x1 = [], []
    n_done = 0
    print(f'Generating {n_samples} reflow pairs...')
    with torch.no_grad():
        with tqdm(total=n_samples, colour="#0055ee") as pbar:
            while n_done < n_samples:
                bs = min(batch_size, n_samples - n_done)
                x0 = torch.randn(bs, 1, 28, 28, device=device)
                x1 = _euler(net, x0.clone(), n_steps, device)
                all_x0.append(x0.cpu())
                all_x1.append(x1.cpu())
                n_done += bs
                pbar.update(bs)

    return torch.cat(all_x0), torch.cat(all_x1)


def reflow(n_epochs, batch_size=128, n_reflow_samples=60000, ode_steps=100,
           time_emb_dim=256, r1_path=R1_PATH, model_path=R2_PATH):
    """
    Round 2: generate coupled pairs via the Round-1 ODE, then retrain.
    The new model learns a straighter velocity field that needs fewer inference steps.
    """
    device = get_device()
    flow_matcher = FlowMatcher()

    x0_data, x1_data = generate_reflow_pairs(
        n_samples=n_reflow_samples, n_steps=ode_steps,
        batch_size=256, time_emb_dim=time_emb_dim, r1_path=r1_path
    )
    dataset = CoupledDataset(x0_data, x1_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    net = _make_net(device, time_emb_dim)
    optim = torch.optim.Adam(net.parameters())

    net.train()
    with tqdm(range(n_epochs), colour="#00ee00") as epoch_pbar:
        for _ in epoch_pbar:
            with tqdm(dataloader, leave=False, colour="#005500") as batch_pbar:
                for x0, x1 in batch_pbar:
                    x0, x1 = x0.to(device), x1.to(device)
                    n = len(x1)
                    t = torch.rand(n, device=device)
                    xt, velocity = flow_matcher.interpolate_coupled(x0, x1, t)
                    loss = nn.functional.mse_loss(net(xt, t), velocity)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    batch_pbar.set_description(f'{loss.item():.3f}')
    torch.save(net.state_dict(), model_path)


##################################################################################################################################
import matplotlib.pyplot as plt


def show_images(images):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(4, 4))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

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


def predict(n_samples=16, c=1, h=28, w=28, n_steps=100, time_emb_dim=256):
    """Euler ODE integration. Prefers Round-2 model; falls back to Round-1."""
    model_path = R2_PATH if os.path.exists(R2_PATH) else R1_PATH
    print(f'Using {model_path}')
    device = get_device()
    net = _make_net(device, time_emb_dim)
    net.load_state_dict(torch.load(model_path, map_location=device))

    net.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, c, h, w, device=device)
        x = _euler(net, x, n_steps, device)
    show_images(x)


def predict_fast(fast_steps=10, n_samples=16, c=1, h=28, w=28, time_emb_dim=256):
    """
    Few-step generation. After reflow the velocity field is much straighter,
    so 1-10 steps often give good results — the main payoff of Rectified Flow.
    """
    predict(n_samples=n_samples, c=c, h=h, w=w, n_steps=fast_steps, time_emb_dim=time_emb_dim)


##################################################################################################################################
from absl import flags
from absl import app


def main(unused_args):
    """
    Samples:
      python rectified_flow_enh.py --train --epochs 100
      python rectified_flow_enh.py --reflow --epochs 100
      python rectified_flow_enh.py --predict --fast
    """
    if FLAGS.train:
        train(n_epochs=FLAGS.epochs)

    if FLAGS.reflow:
        reflow(n_epochs=FLAGS.epochs)

    if FLAGS.predict:
        if FLAGS.fast:
            predict_fast()
        else:
            predict()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, "Round 1 training with independent coupling")
    flags.DEFINE_bool("reflow", False, "Generate coupled pairs and train Round 2")
    flags.DEFINE_bool("predict", False, "Generate images")
    flags.DEFINE_integer("epochs", 3, "Epochs to train")
    flags.DEFINE_bool("fast", False, "Fewer ODE steps (payoff is larger after reflow)")

    app.run(main)
