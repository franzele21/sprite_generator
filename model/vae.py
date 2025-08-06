import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from pytorch_msssim import SSIM


# Assuming Encoder and Decoder are defined in submodel.py
try:
    from .submodel import Encoder, Decoder 
except:
    from submodel import Encoder, Decoder 

import time

class SIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SIM_Loss, self).forward(img1, img2) )


class CSVDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 nrows: int,
                 transform=None,
                 chunk_size=10000):
        
        self.transform = transform
        self.nrows = nrows
        self.chunk_size = chunk_size
        self.csv_path = csv_path
        
        # Load data more efficiently
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the CSV data"""
        print(f"Loading {self.nrows} samples from {self.csv_path}...")
        
        # Read the data directly, sampling if needed
        if self.nrows > 0:
            # Use skiprows to randomly sample from the file
            total_rows = sum(1 for _ in open(self.csv_path, 'r')) - 1  # Subtract header
            skip_indices = sorted(np.random.choice(
                range(1, total_rows + 1), 
                size=max(0, total_rows - self.nrows), 
                replace=False
            ))
            
            # Read CSV with pixel columns only (assuming columns 3 to 3+64*64)
            self.pixel_data = pd.read_csv(
                self.csv_path,
                usecols=list(range(3, 3 + 64*64)),
                skiprows=skip_indices,
                header=0
            )
        else:
            # Load all data
            self.pixel_data = pd.read_csv(
                self.csv_path,
                usecols=list(range(3, 3 + 64*64)),
                header=0
            )
        
        # Convert to tensor and normalize
        self.pixel_data = torch.tensor(self.pixel_data.values, dtype=torch.float32)
        self.num_samples = len(self.pixel_data)
        
        print(f"Loaded {self.num_samples} samples with shape {self.pixel_data.shape}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single sample and reshape to image format"""
        # Get flattened pixel data
        pixel_data = self.pixel_data[idx]
        
        # Reshape to 64x64 image
        image_tensor = pixel_data.view(64, 64)
        
        # Normalize to [0, 1]
        if image_tensor.max() > 0:
            image_tensor = image_tensor / image_tensor.max()
        
        # Add channel dimension: (1, 64, 64)
        image_tensor = image_tensor.unsqueeze(0)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor


class VAE(nn.Module):
    def __init__(self, 
                 conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...],
                 reparam_size: int,
                 num_resnet_blocks: int,
                 expansion_factor: int,
                 rand_intensity: float=0.5,
                 original_image_dims: tuple[int, int]=(64,64)):  # (H, W)
        super().__init__()
        
        self.encoder = Encoder(
            conv_layers_encoder_config, 
            mlp_layers_encoder_config, 
            reparam_size,
            num_resnet_blocks,
            randomize=True,
            return_mean_logvar=True,
            rand_intensity=rand_intensity
        )

        encoder_conv_output_shape = self.encoder.conv_output_size.shape
        
        self.decoder = Decoder(
            conv_layers_encoder_config, 
            mlp_layers_encoder_config, 
            reparam_size,
            num_resnet_blocks,
            expansion_factor,
            encoder_conv_output_shape,
            original_image_dims,
        )
        
        # Store image dimensions
        self.img_c = conv_layers_encoder_config[0][0]
        self.img_h, self.img_w = original_image_dims

        self.reconstruction_loss = SIM_Loss(data_range=1.0, size_average=True, channel=1)
        
        print(f"VAE initialized for images: {self.img_c}x{self.img_h}x{self.img_w}")
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and decoder"""
        z, mean, log_var, resnet_output = self.encoder(x)
        reconstructed_x = self.decoder(z, resnet_output)
        return reconstructed_x, mean, log_var
    
    def vae_loss(self, recon_x, x, mu, log_var, kld_weight=1.0):
        """Calculate VAE loss (reconstruction + KL divergence)"""
        batch_size = x.size(0)
        
        # Reconstruction loss (per sample)
        recon_loss = F.l1_loss(recon_x, x, reduction='sum') / batch_size
        recon_loss += self.reconstruction_loss(recon_x, x) / batch_size
        
        # KL divergence loss (per sample)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        
        # Total loss
        total_loss = recon_loss + kld_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_model(self, 
                    data_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    epochs: int, 
                    device: torch.device,
                    kld_weight: float = 1.0):
        """Train the VAE model"""
        self.train()
        self.to(device)
        
        print(f"Starting training on {device} for {epochs} epochs...")
        print(f"Expected image shape: ({self.img_c}, {self.img_h}, {self.img_w})")

        progress_bar = tqdm(range(epochs))
  
        for epoch in progress_bar:
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            num_batches = 0
            
            for batch_data in data_loader:
                # Handle different data formats
                if isinstance(batch_data, (list, tuple)):
                    images = batch_data[0]
                else:
                    images = batch_data
                images = images.to(device)
                
                optimizer.zero_grad()
                recon_images, mu, log_var = self(images)
                
                loss, recon_loss, kl_loss = self.vae_loss(
                    recon_images, images, mu, log_var, kld_weight
                )
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'KL': f'{kl_loss.item():.4f}'
                })
            
    def generate(self, num_samples=1, device=None):
        """Generate new samples from the latent space"""
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.encoder.reparam_size).to(device)
            generated = self.decoder(z)
        
        return generated
    
    def reconstruct(self, x):
        """Reconstruct input images"""
        self.eval()
        with torch.no_grad():
            recon_x, _, _ = self(x)
        return recon_x
    
    def save(self, path="model/vae.pth"):
        """Save model state"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path="model/vae.pth", device=None):
        """Load model state"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")


# Example usage and testing
if __name__ == "__main__":
    from torchsummary import summary

    # Configuration
    enc_conv_layers = (
        # (in_ch, out_ch, kernel, stride, padding)
        (1, 26, 28, 8, 4),
        (26, 22, 4, 4, 5),
        (22, 16, 5, 5, 1)
    )
    
    enc_mlp_layers = (256, 256, 256, 256, 256)
    latent_size = 256
    encoder_final_conv_shape = (32, 2, 2)
    original_img_dims = (64, 64)
    num_resnet_blocks = 2

    
    # Create VAE
    vae = VAE(
        conv_layers_encoder_config=enc_conv_layers,
        mlp_layers_encoder_config=enc_mlp_layers,
        reparam_size=latent_size,
        num_resnet_blocks=num_resnet_blocks,
        expansion_factor=3,
        original_image_dims=original_img_dims,
        rand_intensity=0.2
    )
    
    # Print model info
    # summary(vae, (1, 64, 64))


    dataset = CSVDataset("./sprites.csv", 5000)  # Smaller sample for testing
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0005)
    
    # Train
    vae.train_model(
        data_loader=train_loader,
        optimizer=optimizer,
        epochs=200,
        device=device,
        kld_weight=0.7
    )
    
    # Save model
    vae.save(f"model/vae_{time.strftime('%a_%d_%Hh_%Mm')}.pth")
    
    print("Training completed successfully!")
