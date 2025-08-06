import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
from model.vae import *
import time
from torch.utils.data import DataLoader


def create_dataloader():
    dummy_dataset = CSVDataset(
        "./sprites.csv",
        40
    )
    train_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=True, num_workers=0) # num_workers > 0 for parallel loading

    return train_loader

def make_images(vae, train_loader):
    train_iter = iter(train_loader)

    f, ax = plt.subplots(4, 2)
    for y in range(4):
        tmp_img = next(train_iter)
        ax[y][0].imshow(tmp_img.reshape((64, 64, 1)))
        ax[y][1].imshow(vae(tmp_img)[0].detach().numpy().reshape((64, 64, 1)))


    plt.savefig(f"result_{time.ctime().replace(' ' , '_')}")
if __name__ == "__main__":
    # --- VAE Configuration ---
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

    vae.load("model/vae_Tue_24_17h_26m.pth")
    vae.eval()

    train_loader = create_dataloader()

    make_images(vae, train_loader)

