from dataclasses import dataclass
import math
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# pip install torch torchvision matplotlib 

# --------------------------- Config ---------------------------------

@dataclass
class Config:
    """Simple configuration container. Modify values here for experiments."""
    seed: int = 69
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_root: str = "./data"
    batch_size: int = 64
    lr: float = 2e-4
    epochs: int = 5
    timesteps: int = 200  # number of diffusion steps (small for speed)
    img_size: int = 28
    in_channels: int = 1
    save_model_path: str = "./diffusion_output/simple_ddpm_mnist_1.pt"
    log_interval: int = 100
    sample_every: int = 1  # sample once per epoch
    num_samples: int = 16


cfg = Config()


# --------------------------- Utilities ---------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


def make_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Create a linear beta schedule from beta_start to beta_end.

    Args:
        timesteps: number of timesteps T
        beta_start: starting beta (near-zero)
        beta_end: final beta

    Returns:
        betas: tensor of shape [T]
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# Precompute schedule tensors (on CPU; we'll move to device later)
betas = make_beta_schedule(cfg.timesteps)
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)
alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), alpha_cumprod[:-1]])

# helper to index tensors by batch of timesteps

def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]):
    """Extract values from a 1-D tensor `a` at indices `t` and reshape to [b, 1, 1, 1]
    so they can broadcast against an image tensor x with shape `x_shape`.

    This is a standard helper used in diffusion implementations.
    """
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t).float() # Move 'a' to the same device as 't'
    return out.view(batch_size, *((1,) * (len(x_shape) - 1)))


# --------------------------- Model ---------------------------------

class SinusoidalPosEmb(nn.Module):
    """Create sinusoidal timestep embeddings (like Transformers).

    This turns an integer timestep t into a vector of dimension dim.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:  # zero pad
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=device)], dim=-1)
        return emb


class SimpleConvBlock(nn.Module):
    """Small convolutional block used in the simple UNet.

    Structure: Conv2d -> BatchNorm -> SiLU -> Conv2d -> BatchNorm -> SiLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class SimpleUNet(nn.Module):
    """A compact UNet-like model for MNIST.

    It accepts a noisy image x_t and the timestep t (as integer tensor), embeds
    t and conditions convolutional layers with that embedding by addition.

    The model predicts the noise (epsilon) added to x_0 to obtain x_t. This is
    the common training objective in DDPM.
    """

    def __init__(self, in_ch=1, base_ch=32, time_emb_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # encoder
        self.enc1 = SimpleConvBlock(in_ch, base_ch)
        self.enc2 = SimpleConvBlock(base_ch, base_ch * 2)

        # bottleneck
        self.mid = SimpleConvBlock(base_ch * 2, base_ch * 2)

        # decoder
        self.dec2 = SimpleConvBlock(base_ch * 4, base_ch)
        self.dec1 = SimpleConvBlock(base_ch * 2, base_ch)

        self.out = nn.Conv2d(base_ch, in_ch, kernel_size=1)

        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # map time embedding to channels so it can be added
        self.time_mlp = nn.Linear(time_emb_dim, base_ch * 2)

    def forward(self, x, t):
        """Forward pass.

        Args:
            x: [B, C, H, W] noisy images at timestep t
            t: [B] integer timesteps

        Returns:
            predicted noise (epsilon) with same shape as x
        """
        # time embedding
        t_emb = self.time_emb(t)  # [B, time_emb_dim]
        t_cond = self.time_mlp(t_emb)[:, :, None, None]  # [B, base_ch*2, 1, 1]

        # encoder
        e1 = self.enc1(x)  # [B, base_ch, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, base_ch*2, H/2, W/2]

        # add time conditioning (broadcasted)
        e2 = e2 + t_cond

        # bottleneck
        m = self.mid(self.pool(e2))  # [B, base_ch*2, H/4, W/4]

        # decode
        u2 = self.upsample(m)  # up to H/2
        u2 = torch.cat([u2, e2], dim=1)  # concat skip
        d2 = self.dec2(u2)

        u1 = self.upsample(d2)  # up to H
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.out(d1)
        return out


# --------------------------- Diffusion helpers ---------------------------------

class Diffusion:
    """Container for diffusion forward/backward utilities.

    This class holds precomputed schedule tensors and provides functions for
    sampling q(x_t | x_0), predicting x_{t-1} during sampling, and a helper to
    run the full reverse diffusion chain.
    """

    def __init__(self, betas: torch.Tensor):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])
        # Calculations often used in the formulas
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Diffuse (add noise) to x_start at step t: q(x_t | x_0).

        If `noise` is None, we sample standard normal noise.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = _extract(self.sqrt_alpha_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = _extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_mean_variance(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor):
        """Compute the mean and variance of p(x_{t-1} | x_t) using the model's prediction.

        The model predicts the noise epsilon_theta; we use the known posterior formula
        from the DDPM paper. This returns the posterior mean and variance as tensors.
        """
        betas_t = _extract(self.betas, t, x_t.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t, x_t.shape)
        alpha_cumprod_t = _extract(self.alpha_cumprod, t, x_t.shape)
        alpha_cumprod_prev_t = _extract(self.alpha_cumprod_prev, t, x_t.shape)

        # predict noise
        eps_theta = model(x_t, t)

        # predicted x_0
        x0_pred = (x_t - _extract(self.sqrt_one_minus_alpha_cumprod, t, x_t.shape) * eps_theta) / _extract(self.sqrt_alpha_cumprod, t, x_t.shape)

        # clip x0_pred to [-1,1] to stabilize (optional)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # posterior mean calculation (DDPM Eq. 4/5)
        posterior_mean = (
            betas_t * x0_pred * (_extract(self.alpha_cumprod_prev, t, x_t.shape).sqrt()) / (1.0 - alpha_cumprod_t)
            + (_extract(self.alphas, t, x_t.shape).sqrt() * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)) * x_t
        )

        # posterior variance (from paper) -- we use the formula for posterior variance
        posterior_variance = betas_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
        return posterior_mean, posterior_variance


# helper functions used above (to avoid recomputing in p_mean_variance)

def sqrt_one_minus_alpha_cumprod_t(x, t):
    return _extract(torch.sqrt(1.0 - alpha_cumprod), t, x.shape)


def alphas_t(x, t):
    return _extract(alphas, t, x.shape)

# slightly simplified sampling step that uses
# the formula from Ho et al. (predict x0 and then compute mean & add noise).


def p_sample_simple(model, diffusion: Diffusion, x_t: torch.Tensor, t: int, device: str):
    """Single step of the reverse diffusion using a simplified posterior.

    Args:
        model: epsilon-predicting network
        diffusion: Diffusion instance
        x_t: current noisy image batch at timestep t
        t: integer timestep scalar (0-based index)
        device: device string

    Returns:
        x_{t-1}
    """
    B = x_t.shape[0]
    t_batch = torch.full((B,), t, dtype=torch.long, device=device)

    betas_t = diffusion.betas[t].to(device)
    alpha_t = diffusion.alphas[t].to(device)
    alpha_cumprod_t = diffusion.alpha_cumprod[t].to(device)
    alpha_cumprod_prev_t = diffusion.alpha_cumprod_prev[t].to(device)

    # model predicts noise
    eps_theta = model(x_t, t_batch)

    # predict x0
    coef1 = 1.0 / math.sqrt(alpha_cumprod_t)
    coef2 = math.sqrt(1 - alpha_cumprod_t)
    x0_pred = coef1 * x_t - (coef2 / math.sqrt(alpha_cumprod_t)) * eps_theta
    x0_pred = x0_pred.clamp(-1.0, 1.0)

    # posterior mean (DDPM closed-form)
    posterior_mean = (
        (alpha_cumprod_prev_t.sqrt() * betas_t / (1.0 - alpha_cumprod_t)) * x0_pred
        + (alpha_t.sqrt() * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)) * x_t
    )

    posterior_variance = betas_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
    if t > 0:
        noise = torch.randn_like(x_t)
        return posterior_mean + torch.sqrt(posterior_variance) * noise
    else:
        return posterior_mean


def sample_loop(model, diffusion: Diffusion, shape: Tuple[int], device: str, verbose: bool = True):
    """Run the full reverse diffusion to sample new images.

    Args:
        model: trained model
        diffusion: Diffusion instance
        shape: (B, C, H, W)
        device: device string

    Returns:
        tensor of generated images in [-1, 1]
    """
    model.eval()
    B = shape[0]
    img = torch.randn(shape, device=device)
    T = diffusion.betas.shape[0]
    with torch.no_grad():
        for t in reversed(range(T)):
            if verbose and t % 50 == 0:
                print(f"Sampling step {t}/{T}")
            img = p_sample_simple(model, diffusion, img, t, device)
    return img


# --------------------------- Dataset & Dataloader ---------------------------------

def get_dataloader(cfg: Config):
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # map to [-1, 1]
    ])
    train = torchvision.datasets.MNIST(cfg.dataset_root, train=True, download=True, transform=transform)
    dl = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return dl


# --------------------------- Training ---------------------------------


def train(cfg: Config):
    device = torch.device(cfg.device)
    dataloader = get_dataloader(cfg)
    model = SimpleUNet(in_ch=cfg.in_channels).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    diffusion = Diffusion(betas)

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            B = x.shape[0]

            # sample random timesteps for each example in the batch
            t = torch.randint(0, cfg.timesteps, (B,), device=device).long()

            # sample noise and create x_t
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise=noise)

            # predict the noise
            eps_theta = model(x_t, t)

            loss = nn.functional.mse_loss(eps_theta, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if global_step % cfg.log_interval == 0:
                print(f"Epoch {epoch} step {global_step} loss: {loss.item():.6f}")

            global_step += 1

        # sample images at end of each epoch
        if (epoch + 1) % cfg.sample_every == 0:
            with torch.no_grad():
                samples = sample_loop(model, diffusion, (cfg.num_samples, cfg.in_channels, cfg.img_size, cfg.img_size), device)
                visualize_samples(samples, epoch, prefix="./diffusion_output/sample_epoch")

    # save the model
    torch.save(model.state_dict(), cfg.save_model_path)
    print(f"Model saved to {cfg.save_model_path}")
    return model


# --------------------------- Visualization ---------------------------------


def visualize_samples(x: torch.Tensor, epoch: int, prefix: str = "sample"):
    """Plot a grid of images (x in [-1,1])."""
    x = x.cpu()
    grid = torchvision.utils.make_grid((x + 1) / 2.0, nrow=int(math.sqrt(x.shape[0])))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.title(f"{prefix} epoch={epoch}")
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    out_name = f"{prefix}_epoch{epoch}.png"
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()
    print(f"Saved samples to {out_name}")


def visualize_reconstructions(model: nn.Module, diffusion: Diffusion, dataset_loader: DataLoader, device: str, n: int = 8):
    """Show some original -> noisy -> reconstructed triplets from the model."""
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(dataset_loader))
        x = x[:n].to(device)
        B = x.shape[0]
        t = torch.randint(0, cfg.timesteps, (B,), device=device)
        noise = torch.randn_like(x)
        x_t = diffusion.q_sample(x, t, noise=noise)
        eps_pred = model(x_t, t)
        # predict x0 as in training sampling helper
        coef1 = 1.0 / torch.sqrt(_extract(alpha_cumprod, t, x.shape))
        coef2 = torch.sqrt(1.0 - _extract(alpha_cumprod, t, x.shape))
        x0_pred = coef1 * x_t - (coef2 / torch.sqrt(_extract(alpha_cumprod, t, x.shape))) * eps_pred
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # convert to [0,1] for visualization
        orig = (x.cpu() + 1) / 2.0
        noisy = (x_t.cpu() + 1) / 2.0
        recon = (x0_pred.cpu() + 1) / 2.0

        # create a grid of triplets
        rows = []
        for i in range(B):
            rows.append(orig[i])
            rows.append(noisy[i])
            rows.append(recon[i])
        grid = torchvision.utils.make_grid(rows, nrow=3)
        plt.figure(figsize=(6, 2 * B))
        plt.axis('off')
        plt.title('original | noisy | reconstruction (per column)')
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
        out_name = './diffusion_output/reconstructions.png'
        plt.savefig(out_name, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstructions to {out_name}")


# --------------------------- Main ---------------------------------

if __name__ == '__main__':
    print("Configuration:\n", cfg)
    device = torch.device(cfg.device)

    dl = get_dataloader(cfg)
    trained_model = train(cfg)

    model = SimpleUNet(in_ch=cfg.in_channels).to(device)
    model.load_state_dict(torch.load(cfg.save_model_path, map_location=device))

    diffusion = Diffusion(betas)
    samples = sample_loop(model, diffusion, (cfg.num_samples, cfg.in_channels, cfg.img_size, cfg.img_size), cfg.device)
    visualize_samples(samples, epoch=cfg.epochs, prefix='final_samples')

    # show reconstruction examples
    visualize_reconstructions(model, diffusion, dl, cfg.device, n=12)

    print("Done.")