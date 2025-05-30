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
from .submodel import Encoder, Decoder 

from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

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
        image_tensor /= image_tensor.max()

        image_tensor = image_tensor.view(1, 64, 64)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor


class VAE(nn.Module):
    def __init__(self, 
                 conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...],
                 reparam_size: int,
                 encoder_conv_output_shape: tuple[int, int, int], # (Channels, Height, Width)
                 original_image_dims: tuple[int, int] # (Height, Width)
                ):
        super().__init__()

        self.encoder = Encoder(conv_layers_encoder_config, 
                               mlp_layers_encoder_config, 
                               reparam_size,
                               randomize=True, # Ensure encoder can sample during training
                               return_mean_logvar=True,
                               rand_intensity=0.5) # Crucial for VAE loss
        self.decoder = Decoder(conv_layers_encoder_config, 
                               mlp_layers_encoder_config, 
                               reparam_size,
                               encoder_conv_output_shape,
                               original_image_dims)
        
        # Store image properties for potential reshaping in the training loop
        # Input channels are from the first convolutional layer's input_channels config
        self.img_c = conv_layers_encoder_config[0][0] 
        self.img_h, self.img_w = original_image_dims

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.
        Args:
            x (torch.Tensor): Input tensor (batch of images).
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - reconstructed_x: The VAE's reconstruction of the input.
                - mean: The mean of the latent distribution.
                - log_var: The log variance of the latent distribution.
        """
        z, mean, log_var = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, log_var

    def train_model(self, 
                    data_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    epochs: int, 
                    device: torch.device,
                    kld_weight: float = 1.0):
        """
        Trains the VAE model.
        Args:
            data_loader (DataLoader): DataLoader providing batches of data.
                                      It's assumed the DataLoader and its Dataset handle
                                      reading from CSVs that may not fit in memory.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            epochs (int): Number of epochs to train for.
            device (torch.device): The device (e.g., 'cuda' or 'cpu') to train on.
            kld_weight (float): Weight for the KL divergence term in the loss (beta in beta-VAE).
        """
        self.train() # Set the VAE model to training mode

        print(f"Starting training on {device} for {epochs} epochs...")

        for epoch in tqdm(range(epochs)):
            total_epoch_loss = 0.0
            total_mse_loss = 0.0
            total_kld_loss = 0.0

            for batch_idx, data_batch in enumerate(data_loader):
                # Ensure data_batch provides image tensors.
                # If data_batch is a tuple/list (e.g., [images, labels]), take the images.
                if isinstance(data_batch, (list, tuple)):
                    images = data_batch[0].to(device)
                else:
                    images = data_batch.to(device)

                # if images.ndim == 2: # e.g., (batch_size, flattened_features)
                #     try:
                #         images = images.view(-1, self.img_c, self.img_h, self.img_w)
                #     except RuntimeError as e:
                #         print(f"Error reshaping images: {e}. Expected C,H,W: ({self.img_c},{self.img_h},{self.img_w}). Got: {images.shape}")
                #         continue
                # elif images.ndim == 3: # Potentially (batch_size, H, W) if channels=1
                #     if self.img_c == 1 and images.shape[0] !=1 : # (B, H, W)
                #          images = images.unsqueeze(1) # Add channel dim: (B, 1, H, W)
                #     elif images.shape[0] == self.img_c : # (C,H,W) from a single sample dataset - unlikely for batch
                #          images = images.unsqueeze(0) # Add batch dim
                #     else:
                #         print(f"Warning: Image batch has 3 dimensions {images.shape} but expected {self.img_c} channels or a clear way to reshape.")
                #         # continue # Or attempt reshape if logic is clear
                # # Ensure 4D: (B, C, H, W)
                # if images.ndim != 4 or images.shape[1] != self.img_c or images.shape[2] != self.img_h or images.shape[3] != self.img_w:
                #     print(f"Warning: Batch images have unexpected shape {images.shape}. Expected ({images.shape[0]},{self.img_c},{self.img_h},{self.img_w}). Skipping batch.")
                #     continue

                optimizer.zero_grad()
                
                reconstructed_x, mean, log_var = self(images) # VAE forward pass
                
                # 1. Reconstruction Loss (MSE)
                # Sum over all pixels and batch elements.
                mse = F.mse_loss(reconstructed_x, images, reduction='sum')
                
                # 2. KL Divergence
                # KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
                kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                
                # Total VAE Loss (ELBO)
                loss = mse + kld_weight * kld # Apply KLD weight
                
                loss.backward()
                optimizer.step()
                
                total_epoch_loss += loss.item()
                total_mse_loss += mse.item()
                total_kld_loss += (kld_weight * kld.item()) # Store weighted KLD
                
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(data_loader):
                    # Average loss per sample in the current batch for reporting
                    avg_batch_loss = loss.item() / images.size(0)
                    avg_batch_mse = mse.item() / images.size(0)
                    avg_batch_kld = (kld_weight * kld.item()) / images.size(0)
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], "
                          f"Avg Batch Loss: {avg_batch_loss:.4f} "
                          f"(MSE: {avg_batch_mse:.4f}, KLD: {avg_batch_kld:.4f})")
            
            # Calculate average loss per sample for the entire epoch
            # Note: len(data_loader.dataset) might not be available for IterableDataset.
            # If so, average over number of batches: num_samples_processed = len(data_loader) * batch_size (approx)
            num_samples_in_epoch = len(data_loader.dataset) if hasattr(data_loader.dataset, '__len__') else (batch_idx + 1) * images.size(0)
            avg_loss = total_epoch_loss / num_samples_in_epoch
            avg_mse = total_mse_loss / num_samples_in_epoch
            avg_kld = total_kld_loss / num_samples_in_epoch # KLD here is already weighted sum
            print(f"Epoch [{epoch+1}/{epochs}] COMPLETED. Avg Loss: {avg_loss:.4f} "
                  f"(Avg MSE: {avg_mse:.4f}, Avg Weighted KLD: {avg_kld:.4f})")
        
        print("--- Training finished ---")
    
    def save(self, path="model/vae.pth"):
        torch.save(self.state_dict(), path)
    
    def load(self, path="model/vae.pth"):
        self.load_state_dict(torch.load(path))

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
    num_epochs = 50 # Keep small for a quick demo
    kld_beta = 1.0 # Weight for the KL divergence term

    # --- Start Training ---
    vae.train_model(
        data_loader=train_loader, 
        optimizer=optimizer, 
        epochs=num_epochs, 
        device=device,
        kld_weight=kld_beta
    )

    print("Example training run completed.")

    vae.save()