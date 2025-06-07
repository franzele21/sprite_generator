"""
TODO:
    - le décodeur fait des gros blocs (voir l'image en root)
        - raffiner le upscaling ?
        - faire une structure en entonnoir sur les dernières couches ?
        - ResNet ?
    - les colonnes du bas et de droite n'ont pas la même couleur
        - dans le upscaling y'a un pb je pense

    - Trouver un VAE qui ressort une bonne reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader # For Dataset and DataLoader
import pandas as pd # Example for CSV handling in Dataset, not directly used by train_model

# Assuming Encoder and Decoder are defined in submodel.py
try:
    from .submodel import Encoder, Decoder 
except:
    from submodel import Encoder, Decoder 

from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import time 

class CSVDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 nrows: int,
                 transform=None,
                 chunk_size=10000):

        self.transform = transform
        self.pixel_data = []
        self.nrows = nrows
        self.chunk_size = chunk_size
        self.csv_path = csv_path

        # Calculate the total number of rows in the CSV file
        self.total_rows = sum(1 for _ in open(csv_path, 'r')) - 1  # Subtract 1 for the header

        # Ensure nrows does not exceed the total number of rows
        self.nrows = min(self.nrows, self.total_rows)

        # Sample row indices
        self.sampled_indices = np.random.choice(self.total_rows, self.nrows, replace=False)

        # Load the sampled rows
        self._load_sampled_rows()

    def _load_sampled_rows(self):
        # Read the CSV file in chunks and collect the sampled rows
        for chunk in tqdm(pd.read_csv(
            self.csv_path,
            usecols=list(range(3, 3+64*64)),
            header=0,
            chunksize=self.chunk_size
        )):
            # Find the intersection of sampled indices and the current chunk
            chunk_start = len(self.pixel_data)
            chunk_end = chunk_start + len(chunk)
            chunk_indices = [i - chunk_start for i in self.sampled_indices if chunk_start <= i < chunk_end]

            if chunk_indices:
                sampled_chunk = chunk.iloc[chunk_indices]
                self.pixel_data.append(sampled_chunk)

        # Concatenate all sampled chunks into a single DataFrame
        self.pixel_data = pd.concat(self.pixel_data, ignore_index=True)

        # Convert to tensor
        self.pixel_data = torch.tensor(self.pixel_data.values, dtype=torch.float32)
        self.num_samples = len(self.pixel_data)  # Number of rows loaded

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a sample (image tensor) from the dataset at the given index.
        """
        flattened_image_data = self.pixel_data[idx]

        image_tensor = flattened_image_data.flatten()
        # Will create a image only with 0 and 1
        # Here, whenever a pixel isn't 0, it will be 1
        # image_tensor[image_tensor>0] = 1
	# Here we round the pixel to the nearest value (either 0 or 1)
        # image_tensor = torch.round(image_tensor/image_tensor.max())

	# Normal/base case
        image_tensor /= image_tensor.max()

        image_tensor = image_tensor.view(1, 64, 64)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor


class VAE(nn.Module):
    def __init__(self, 
                 n_encoders: int,
                 conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...],
                 reparam_size: int,
                 encoder_conv_output_shape: tuple[int, int, int],
                 original_image_dims: tuple[int, int]):
        super().__init__()

        self.encoder = Encoder(n_encoders,
                               conv_layers_encoder_config, 
                               mlp_layers_encoder_config, 
                               reparam_size,
                               randomize=True,
                               return_mean_logvar=True)
                               
        self.decoder = Decoder(conv_layers_encoder_config, 
                               mlp_layers_encoder_config, 
                               reparam_size,
                               encoder_conv_output_shape,
                               original_image_dims)
        
        self.img_c = conv_layers_encoder_config[0][0] 
        self.img_h, self.img_w = original_image_dims

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mean, log_var = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, log_var

    def kl_divergence(self, mean, logvar):
        """
        Compute KL divergence between learned distribution and standard normal
        KL(q(z|x) || p(z)) where p(z) = N(0,I)
        """
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)

    def train_model(self, 
                    data_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    epochs: int, 
                    device: torch.device,
                    beta_start: float = 0.0,
                    beta_end: float = 1.0,
                    beta_warmup_epochs: int = 10):
        """
        Train with beta annealing for better KL divergence control
        """
        self.train()
        print(f"Starting training on {device} for {epochs} epochs...")

        for epoch in tqdm(range(epochs), total=epochs):
            # Beta annealing - gradually increase KL weight
            if epoch < beta_warmup_epochs:
                beta = beta_start + (beta_end - beta_start) * (epoch / beta_warmup_epochs)
            else:
                beta = beta_end

            total_epoch_loss = 0.0
            total_recon_loss = 0.0
            total_kld_loss = 0.0
            num_batches = 0

            for batch_idx, data_batch in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
                if isinstance(data_batch, (list, tuple)):
                    images = data_batch[0].to(device)
                else:
                    images = data_batch.to(device)

                optimizer.zero_grad()
                
                reconstructed_x, mean, log_var = self(images)
                
                # Reconstruction loss (per sample average)
                recon_loss = F.mse_loss(reconstructed_x, images, reduction='mean')
                
                # KL divergence (per sample average)
                kld = self.kl_divergence(mean, log_var).mean()
                
                # Total loss with beta weighting
                loss = recon_loss + beta * kld
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_epoch_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kld_loss += kld.item()
                num_batches += 1
                
                if (batch_idx + 1) % 50 == 0:
                    print(f" Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], "
                          f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                          f"KLD: {kld.item():.4f}, Beta: {beta:.4f}")
            
            # Average losses for the epoch
            avg_loss = total_epoch_loss / num_batches
            avg_recon = total_recon_loss / num_batches
            avg_kld = total_kld_loss / num_batches
            
            print(f"Epoch [{epoch+1}/{epochs}] COMPLETED. "
                  f"Avg Loss: {avg_loss:.4f}, Avg Recon: {avg_recon:.4f}, "
                  f"Avg KLD: {avg_kld:.4f}, Beta: {beta:.4f}")
        
        print("--- Training finished ---")
    
    def save(self, path="model/vae.pth"):
        torch.save(self.state_dict(), path)
    
    def load(self, path="model/vae.pth"):
        self.load_state_dict(torch.load(path, map_location='cpu'))

# Example of how to use it:
if __name__ == "__main__":
    from torchsummary import summary

    # --- VAE Configuration ---
    enc_conv_layers = (
        (1, 16, 16, 8, 0),    # In: (1, 64, 64) -> Out: (16, 7, 7)
        (16, 24, 8, 4, 2),   # In: (16, 7, 7) -> Out: (24, 1, 1)
        (24, 32, 2, 1, 1)    # In: (24, 1, 1) -> Out: (32, 2, 2)
    )
    # Flattened output of conv layers: 32 * 2 * 2 = 128
    enc_mlp_layers = (128, 64, 32) # 128, 128, 128 ?
    latent_size = 32 # 128 ?
    encoder_final_conv_shape = (32, 2, 2) # (Channels, Height, Width)
    original_img_dims = (64, 64)         # (Height, Width)
    input_channels = enc_conv_layers[0][0] # Should be 1

    # --- Instantiate VAE ---
    vae = VAE(
        n_encoders=5,
        conv_layers_encoder_config=enc_conv_layers,
        mlp_layers_encoder_config=enc_mlp_layers,
        reparam_size=latent_size,
        encoder_conv_output_shape=encoder_final_conv_shape,
        original_image_dims=original_img_dims
    )
    print("VAE Model Structure:")
    summary(vae, (input_channels, original_img_dims[0], original_img_dims[1]))
    print("-" * 60)


    # Create dummy data (e.g., 100 samples)
    dummy_dataset = CSVDataset(
        "./sprites.csv",
        50_000
    )
    
    # DataLoader handles batching and can use multiple workers for loading
    # If your Dataset handles large CSVs (e.g., by chunking), DataLoader works with it.
    train_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True, num_workers=0) # num_workers > 0 for parallel loading

    # --- Training Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device) # Move VAE model to the selected device

    optimizer = torch.optim.Adam(vae.parameters(), lr=2e-3)
    num_epochs = 10 # Keep small for a quick demo

    # --- Start Training ---
    vae.train_model(
        data_loader=train_loader, 
        optimizer=optimizer, 
        epochs=num_epochs, 
        device=device,
    )

    print("Example training run completed.")

    vae.save(f"model/vae_{time.strftime('%a_%d_%Hh_%Mm')}.pth")
