"""
This file is created by Antigravity with Gemini and me.
The agent referenced Tianhong's codes in Github.

Just Image Transformer (JiT) with Elucidated Diffusion Models (EDM).
Everything in one file.

This implements the pixel-space large-patch generative model proposed in:
"Back to Basics: Let Denoising Generative Models Denoise" (Tianhong Li & Kaiming He, arXiv:2511.13720)
using the preconditioning, loss weighting, and deterministic sampling framework from:
"Elucidating the Design Space of Diffusion-Based Generative Models" (EDM, Karras et al., arXiv:2206.00364)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Model Architecture (JiT)
# ==============================================================================

class BottleneckPatchEmbed(nn.Module):
    """
    Image to Patch Embedding with a bottleneck CNN structure.
    Projects raw patches using two sequential convolutions to keep efficiency.
    """
    def __init__(self, img_size=28, patch_size=7, in_chans=1, pca_dim=64, embed_dim=128, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # First stage of bottleneck CNN (kernel & stride equal to patch size)
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # Second stage of bottleneck projection (1x1 Conv)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        
        x = self.proj1(x)  # (B, pca_dim, H_patches, W_patches)
        x = self.proj2(x)  # (B, embed_dim, H_patches, W_patches)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class BottleneckUnpatchEmbed(nn.Module):
    """
    Reverse of BottleneckPatchEmbed.
    Projects token embeddings back to raw pixel space.
    """
    def __init__(self, img_size=28, patch_size=7, in_chans=1, pca_dim=64, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches_side = img_size // patch_size

        # Reverse bottleneck stages using linear layers
        self.proj1 = nn.Linear(embed_dim, pca_dim)
        self.proj2 = nn.Linear(pca_dim, in_chans * patch_size * patch_size)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj1(x)  # (B, num_patches, pca_dim)
        x = self.proj2(x)  # (B, num_patches, in_chans * patch_size * patch_size)

        # Unflatten and reshape patches back into grid
        x = x.reshape(B, self.num_patches_side, self.num_patches_side, self.in_chans, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, in_chans, num_patches_side, patch_size, num_patches_side, patch_size)
        x = x.reshape(B, self.in_chans, self.img_size, self.img_size)
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time/noise."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        # t is continuous noise scaling (B,)
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


class JiTBlock(nn.Module):
    """
    Just Image Transformer Block.
    Features:
    - elementwise_affine=False in LayerNorm to support adaptive modulation.
    - adaLN-Zero conditioning to predict scale/shift/gate parameters.
    - SwiGLU feedforward MLP.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.ln2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        # SwiGLU MLP layers
        self.mlp_gate = nn.Linear(embed_dim, 4 * embed_dim)
        self.mlp_in = nn.Linear(embed_dim, 4 * embed_dim)
        self.mlp_out = nn.Linear(4 * embed_dim, embed_dim)
        self.mlp_drop = nn.Dropout(dropout)

        # adaLN-Zero conditioning generator
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim)
        )

        # Initialize modulation projections to zero (making the block initially identity)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        # x: (B, num_patches, embed_dim), c: (B, embed_dim)
        mods = self.adaLN_modulation(c)  # (B, 6 * embed_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)

        # Attention layer path with modulation
        res = self.ln1(x)
        res = res * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        res, _ = self.attn(res, res, res)
        x = x + gate_msa.unsqueeze(1) * res

        # SwiGLU MLP layer path with modulation
        res = self.ln2(x)
        res = res * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        gate = F.silu(self.mlp_gate(res))
        v = self.mlp_in(res)
        res = gate * v
        res = self.mlp_out(res)
        res = self.mlp_drop(res)

        x = x + gate_mlp.unsqueeze(1) * res
        return x


class JiT(nn.Module):
    """
    Just Image Transformer model.
    Bypasses VAE latents; operates directly on large raw pixel patches.
    """
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=128, depth=4, num_heads=4, num_classes=10, bottleneck_dim=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if bottleneck_dim is None:
            bottleneck_dim = embed_dim // 2

        self.patch_embed = BottleneckPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            pca_dim=bottleneck_dim,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Continuous Noise / Time Embeddings
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Class label embeddings (adds class 10 for classifier-free guidance)
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
            nn.init.normal_(self.class_embed.weight, std=0.02)
        else:
            self.class_embed = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            JiTBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        # Final Layer Norm and modulated scaling
        self.ln = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim)
        )
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)

        # Unpatchify projection back to image space
        self.unpatch_embed = BottleneckUnpatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            pca_dim=bottleneck_dim,
            embed_dim=embed_dim
        )

    def forward(self, x, sigma, class_labels=None):
        # x: (B, in_chans, H, W)
        # sigma: (B,)
        # class_labels: (B,) or None

        # 1. Conditioning vector
        # Scale continuous noise level sigma to logarithmic conditioning c_noise
        c_noise = 0.25 * torch.log(sigma + 1e-8)
        c = self.time_embed(c_noise)

        if class_labels is not None and self.class_embed is not None:
            c = c + self.class_embed(class_labels)

        # 2. Patchification and positional embedding
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed

        # 3. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)

        # 4. Final normalization and modulation
        mods = self.final_modulation(c)
        scale, shift = mods.chunk(2, dim=-1)
        tokens = self.ln(tokens)
        tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 5. Unpatchification to image pixels
        out_pixels = self.unpatch_embed(tokens)
        return out_pixels


# ==============================================================================
# EDM Preconditioning Wrapper
# ==============================================================================

class EDMPreconditionedModel(nn.Module):
    """
    Wraps the core neural network and implements EDM preconditioning.
    This guarantees that the network operates on inputs normalized to unit variance
    and predicts the clean target y = f(x) stably across all noise levels.
    """
    def __init__(self, net, sigma_data=0.5):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def forward(self, x, sigma, class_labels=None):
        # x: (B, C, H, W) noised image
        # sigma: (B,) noise level
        # class_labels: (B,) labels

        B = x.shape[0]
        sigma_view = sigma.reshape(B, 1, 1, 1)

        # Preconditioning scaling coefficients
        c_skip = self.sigma_data**2 / (sigma_view**2 + self.sigma_data**2)
        c_out = sigma_view * self.sigma_data / torch.sqrt(sigma_view**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(sigma_view**2 + self.sigma_data**2)

        # Scaled network input
        x_in = c_in * x

        # Run core Transformer model
        net_out = self.net(x_in, sigma, class_labels=class_labels)

        # Preconditioned clean image estimate F_out
        f_out = c_skip * x + c_out * net_out
        return f_out


# ==============================================================================
# Training and Dataset Utility Functions
# ==============================================================================

import torchvision
from tqdm.auto import tqdm


def get_dataloader(batch_size=128):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.mnist.MNIST("./data", download=True, train=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    elif torch.backends.mps.is_available():
        return 'mps:0'
    return 'cpu'


def train(n_epochs, batch_size=128, embed_dim=128, depth=4, num_heads=4, bottleneck_dim=32,
          P_mean=-1.2, P_std=1.2, sigma_data=0.5, label_drop_prob=0.15,
          model_path='jit.pth'):
    device = get_device()
    print(f"Training on device: {device}")
    
    dataloader = get_dataloader(batch_size=batch_size)
    
    # Instantiate core Transformer and EDM preconditioned wrapper
    raw_net = JiT(
        img_size=28,
        patch_size=7,
        in_chans=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_classes=10,
        bottleneck_dim=bottleneck_dim
    )
    model = EDMPreconditionedModel(raw_net, sigma_data=sigma_data).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    model.train()
    with tqdm(range(n_epochs), colour="#00ee00", desc="Epochs") as epoch_pbar:
        for epoch in epoch_pbar:
            running_loss = 0.0
            with tqdm(dataloader, leave=False, colour="#005500", desc="Batches") as batch_pbar:
                for y, labels in batch_pbar:
                    y = y.to(device)
                    labels = labels.to(device)
                    B = len(y)

                    # 1. Sample continuous noise levels sigma from log-normal distribution
                    log_sigma = torch.randn(B, device=device) * P_std + P_mean
                    sigma = torch.exp(log_sigma)

                    # 2. Add noise to original clean image y
                    noise = torch.randn_like(y)
                    sigma_view = sigma.reshape(B, 1, 1, 1)
                    x = y + sigma_view * noise

                    # 3. Label dropout for Classifier-Free Guidance (CFG)
                    # Class label 10 is reserved for unconditioned prediction
                    drop_mask = torch.rand(B, device=device) < label_drop_prob
                    class_labels = torch.where(drop_mask, torch.full_like(labels, 10), labels)

                    # 4. Predict clean image and compute preconditioned weighted loss
                    f_out = model(x, sigma, class_labels=class_labels)

                    # Loss weighting function w(sigma) designed to normalize regression target variance
                    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
                    weight_view = weight.reshape(B, 1, 1, 1)
                    
                    loss = (weight_view * (f_out - y)**2).mean()

                    # 5. Backward pass
                    optim.zero_grad()
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optim.step()

                    running_loss += loss.item()
                    batch_pbar.set_description(f"Loss: {loss.item():.4f}")
            
            epoch_loss = running_loss / len(dataloader)
            epoch_pbar.set_postfix({"Epoch Loss": f"{epoch_loss:.4f}"})

    # Save trained state dictionary
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")


# ==============================================================================
# Sampling and Evaluation (EDM ODE Sampler)
# ==============================================================================

import matplotlib.pyplot as plt


def show_images(images, title="Generated Digits"):
    if isinstance(images, torch.Tensor):
        images = torch.clamp(images, 0.0, 1.0)
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(5, 5))
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < len(images):
                fig.add_subplot(rows, cols, idx + 1)
                plt.imshow(images[idx][0], cmap="gray", vmin=0.0, vmax=1.0)
                plt.axis('off')
                idx += 1
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def predict(n_samples=16, embed_dim=128, depth=4, num_heads=4, bottleneck_dim=32,
            num_steps=50, cfg_scale=2.0, sigma_data=0.5,
            sigma_max=80.0, sigma_min=0.002, rho=7.0,
            model_path='jit.pth'):
    """
    EDM Deterministic ODE Sampler using 2nd-order Heun's Method.
    Steps backwards from maximum noise level sigma_max to sigma_min.
    """
    device = get_device()
    print(f"Generating samples on device: {device}")

    # Load model
    raw_net = JiT(
        img_size=28,
        patch_size=7,
        in_chans=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_classes=10,
        bottleneck_dim=bottleneck_dim
    )
    model = EDMPreconditionedModel(raw_net, sigma_data=sigma_data).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Generate labels for visual grid: digits 0, 1, 2... repeat
    cond_labels = torch.tensor([i % 10 for i in range(n_samples)], device=device)
    uncond_labels = torch.full((n_samples,), 10, dtype=torch.long, device=device)

    # Pre-compute noise schedule levels based on rho exponent
    steps = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigmas = (sigma_max**(1/rho) + steps / (num_steps - 1) * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    # Append final 0.0 at the end
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

    with torch.no_grad():
        # Initialize x at maximum noise level
        x = torch.randn(n_samples, 1, 28, 28, device=device) * sigmas[0]

        for i in tqdm(range(num_steps), desc="ODE Sampling Steps", colour="#0088ff"):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            sigma_tensor = torch.full((n_samples,), sigma, device=device)

            # Evaluate preconditioned model (with CFG support)
            if cfg_scale > 1.0:
                f_out_cond = model(x, sigma_tensor, class_labels=cond_labels)
                f_out_uncond = model(x, sigma_tensor, class_labels=uncond_labels)
                f_out = f_out_uncond + cfg_scale * (f_out_cond - f_out_uncond)
            else:
                f_out = model(x, sigma_tensor, class_labels=cond_labels)

            # Euler step derivative
            d = (x - f_out) / sigma
            x_next = x + (sigma_next - sigma) * d

            # 2nd-order correction step (Heun's method)
            if sigma_next > 0:
                sigma_next_tensor = torch.full((n_samples,), sigma_next, device=device)
                if cfg_scale > 1.0:
                    f_out_cond_next = model(x_next, sigma_next_tensor, class_labels=cond_labels)
                    f_out_uncond_next = model(x_next, sigma_next_tensor, class_labels=uncond_labels)
                    f_out_next = f_out_uncond_next + cfg_scale * (f_out_cond_next - f_out_uncond_next)
                else:
                    f_out_next = model(x_next, sigma_next_tensor, class_labels=cond_labels)

                d_next = (x_next - f_out_next) / sigma_next
                x_next = x + (sigma_next - sigma) * 0.5 * (d + d_next)

            x = x_next

    show_images(x, title=f"JiT + EDM Generated Digits (CFG: {cfg_scale})")


# ==============================================================================
# Script Entry Point
# ==============================================================================

from absl import flags
from absl import app

FLAGS = flags.FLAGS

def main(unused_args):
    """
    Example usage:
      python jit.py --train --epochs 10
      python jit.py --predict --steps 50 --cfg 2.0
    """
    if FLAGS.train:
        train(
            n_epochs=FLAGS.epochs,
            batch_size=FLAGS.batch_size,
            embed_dim=FLAGS.embed_dim,
            depth=FLAGS.depth,
            num_heads=FLAGS.num_heads,
            bottleneck_dim=FLAGS.bottleneck_dim,
            P_mean=FLAGS.P_mean,
            P_std=FLAGS.P_std,
            model_path=FLAGS.model_path
        )

    if FLAGS.predict:
        predict(
            n_samples=FLAGS.n_samples,
            embed_dim=FLAGS.embed_dim,
            depth=FLAGS.depth,
            num_heads=FLAGS.num_heads,
            bottleneck_dim=FLAGS.bottleneck_dim,
            num_steps=FLAGS.steps,
            cfg_scale=FLAGS.cfg,
            model_path=FLAGS.model_path
        )


if __name__ == '__main__':
    flags.DEFINE_bool("train", False, "Train the model")
    flags.DEFINE_bool("predict", False, "Generate digits from noise")
    
    flags.DEFINE_integer("epochs", 100, "Number of training epochs")
    flags.DEFINE_integer("batch_size", 128, "Batch size for training")
    flags.DEFINE_integer("n_samples", 16, "Number of samples to generate")
    flags.DEFINE_integer("steps", 50, "Number of ODE solver steps during generation")
    flags.DEFINE_float("cfg", 10.0, "Classifier-free guidance scale")
    
    # Model architecture size flags
    flags.DEFINE_integer("embed_dim", 128, "Token/embedding dimension of the Transformer")
    flags.DEFINE_integer("bottleneck_dim", 32, "Bottleneck dimension d' of patch projection")
    flags.DEFINE_integer("depth", 4, "Number of transformer layers")
    flags.DEFINE_integer("num_heads", 4, "Number of attention heads")

    # Log-normal noise hyper-parameters (EDM defaults)
    flags.DEFINE_float("P_mean", -1.2, "Log-normal mean for noise sampling distribution during training")
    flags.DEFINE_float("P_std", 1.2, "Log-normal standard deviation for noise sampling distribution during training")
    
    flags.DEFINE_string("model_path", "jit.pth", "Path to save or load the model parameters")

    app.run(main)
