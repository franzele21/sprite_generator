import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
from model.vae import *
import time
from torch.utils.data import DataLoader

# --- VAE Configuration ---
enc_conv_layers = (
    (1, 16, 16, 8, 0),    # In: (1, 64, 64) -> Out: (16, 7, 7)
    (16, 24, 8, 4, 2),   # In: (16, 7, 7) -> Out: (24, 1, 1)
    (24, 32, 2, 1, 1)    # In: (24, 1, 1) -> Out: (32, 2, 2)
)
# Flattened output of conv layers: 32 * 2 * 2 = 128
enc_mlp_layers = (128, 64, 32) # 128, 128, 128 ?
latent_size = 128
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

vae.load("model/vae_rc.pth")
vae.eval()
# Create dummy data (e.g., 100 samples)
dummy_dataset = CSVDataset(
    "./sprites.csv",
    40
)

# DataLoader handles batching and can use multiple workers for loading
# If your Dataset handles large CSVs (e.g., by chunking), DataLoader works with it.
train_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=True, num_workers=0) # num_workers > 0 for parallel loading

train_iter = iter(train_loader)

f, ax = plt.subplots(4, 2)
for y in range(4):
    tmp_img = next(train_iter)
    ax[y][0].imshow(tmp_img.reshape((64, 64, 1)))
    ax[y][1].imshow(vae(tmp_img)[0].detach().numpy().reshape((64, 64, 1)))


plt.savefig(f"result_{time.ctime().replace(' ' , '_').replace(":", "_")}")