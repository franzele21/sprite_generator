import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# Assuming Encoder and EnhancedDecoder are already defined in submodel.py
try:
    from .submodel import Encoder, Decoder
except ImportError:
    from submodel import Encoder, Decoder

class CSVDataset(Dataset):
    def __init__(self, csv_path: str, nrows: int, transform=None, chunk_size=10000):
        self.transform = transform
        self.pixel_data = []
        self.nrows = nrows
        self.chunk_size = chunk_size
        self.csv_path = csv_path

        self.total_rows = sum(1 for _ in open(csv_path, 'r')) - 1
        self.nrows = min(self.nrows, self.total_rows)
        self.sampled_indices = np.random.choice(self.total_rows, self.nrows, replace=False)
        self._load_sampled_rows()

    def _load_sampled_rows(self):
        for chunk in tqdm(pd.read_csv(
            self.csv_path,
            usecols=list(range(3, 3 + 64 * 64)),
            header=0,
            chunksize=self.chunk_size
        )):
            chunk_start = len(self.pixel_data)
            chunk_end = chunk_start + len(chunk)
            chunk_indices = [i - chunk_start for i in self.sampled_indices if chunk_start <= i < chunk_end]

            if chunk_indices:
                sampled_chunk = chunk.iloc[chunk_indices]
                self.pixel_data.append(sampled_chunk)

        self.pixel_data = pd.concat(self.pixel_data, ignore_index=True)
        self.pixel_data = torch.tensor(self.pixel_data.values, dtype=torch.float32)
        self.num_samples = len(self.pixel_data)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        flattened_image_data = self.pixel_data[idx]
        image_tensor = flattened_image_data.flatten()
        image_tensor /= image_tensor.max()
        image_tensor = image_tensor.view(1, 64, 64)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor

class VAE(nn.Module):
    def __init__(self, conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...], reparam_size: int,
                 encoder_conv_output_shape: tuple[int, int, int], original_image_dims: tuple[int, int]):
        super().__init__()

        self.encoder = Encoder(conv_layers_encoder_config, mlp_layers_encoder_config, reparam_size,
                               randomize=True, return_mean_logvar=True, rand_intensity=0.5)
        self.decoder = Decoder(conv_layers_encoder_config, mlp_layers_encoder_config, reparam_size,
                               encoder_conv_output_shape, original_image_dims)

        self.img_c = conv_layers_encoder_config[0][0]
        self.img_h, self.img_w = original_image_dims

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mean, log_var = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, log_var

    def train_model(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer, epochs: int,
                    device: torch.device, kld_weight: float = 1.0):
        self.train()
        print(f"Starting training on {device} for {epochs} epochs...")

        for epoch in tqdm(range(epochs)):
            total_epoch_loss = 0.0
            total_mse_loss = 0.0
            total_kld_loss = 0.0

            for batch_idx, data_batch in enumerate(data_loader):
                if isinstance(data_batch, (list, tuple)):
                    images = data_batch[0].to(device)
                else:
                    images = data_batch.to(device)

                optimizer.zero_grad()
                reconstructed_x, mean, log_var = self(images)
                mse = F.mse_loss(reconstructed_x, images, reduction='sum')
                kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = mse + kld_weight * kld
                loss.backward()
                optimizer.step()

                total_epoch_loss += loss.item()
                total_mse_loss += mse.item()
                total_kld_loss += (kld_weight * kld.item())

                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(data_loader):
                    avg_batch_loss = loss.item() / images.size(0)
                    avg_batch_mse = mse.item() / images.size(0)
                    avg_batch_kld = (kld_weight * kld.item()) / images.size(0)
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], "
                          f"Avg Batch Loss: {avg_batch_loss:.4f} (MSE: {avg_batch_mse:.4f}, KLD: {avg_batch_kld:.4f})")

            num_samples_in_epoch = len(data_loader.dataset) if hasattr(data_loader.dataset, '__len__') else (batch_idx + 1) * images.size(0)
            avg_loss = total_epoch_loss / num_samples_in_epoch
            avg_mse = total_mse_loss / num_samples_in_epoch
            avg_kld = total_kld_loss / num_samples_in_epoch
            print(f"Epoch [{epoch+1}/{epochs}] COMPLETED. Avg Loss: {avg_loss:.4f} "
                  f"(Avg MSE: {avg_mse:.4f}, Avg Weighted KLD: {avg_kld:.4f})")

        print("--- Training finished ---")

    def save(self, path="model/vae_rc.pth"):
        torch.save(self.state_dict(), path)

    def load(self, path="model/vae_rc.pth"):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    from torchsummary import summary

    parser = argparse.ArgumentParser(description="Train a VAE model.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to use for training (default: cuda)")
    args = parser.parse_args()

    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    enc_conv_layers = (
        (1, 16, 16, 8, 0),
        (16, 24, 8, 4, 2),
        (24, 32, 2, 1, 1)
    )
    enc_mlp_layers = (128, 64, 32)
    latent_size = 128
    encoder_final_conv_shape = (32, 2, 2)
    original_img_dims = (64, 64)
    input_channels = enc_conv_layers[0][0]

    vae = VAE(
        conv_layers_encoder_config=enc_conv_layers,
        mlp_layers_encoder_config=enc_mlp_layers,
        reparam_size=latent_size,
        encoder_conv_output_shape=encoder_final_conv_shape,
        original_image_dims=original_img_dims
    ).to(device)  # Move model to device

    print("VAE Model Structure:")
    summary(vae, (input_channels, original_img_dims[0], original_img_dims[1]), device=str(device))
    print("-" * 60)

    dummy_dataset = CSVDataset("./sprites.csv", 50_000)
    train_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(vae.parameters(), lr=2e-3)
    num_epochs = 100
    kld_beta = 1.0

    # vae.train_model(
    #     data_loader=train_loader,
    #     optimizer=optimizer,
    #     epochs=num_epochs,
    #     device=device,
    #     kld_weight=kld_beta
    # )

    print("Example training run completed.")
    # vae.save()
