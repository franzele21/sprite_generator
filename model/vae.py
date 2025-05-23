import torch
import torch.nn as nn
import torch.nn.functional as F

from submodel import *

class VAE(nn.Module):
    def __init__(self, 
                 conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...],
                 reparam_size: int,
                 encoder_conv_output_shape: tuple[int, int, int], # (Channels, Height, Width)
                 original_image_dims: tuple[int, int] # (Height, Width)):
                 ):
        super().__init__()

        self.encoder = Encoder(conv_layers_encoder_config, 
                               mlp_layers_encoder_config, 
                               reparam_size)
        self.decoder = Decoder(conv_layers_encoder_config, 
                               mlp_layers_encoder_config, 
                               reparam_size,
                               encoder_conv_output_shape,
                               original_image_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y
    
if __name__ == "__main__":
    from torchsummary import summary

    enc_conv_layers = (
        (1, 16, 16, 8, 0),
        (16, 24, 8, 4, 2),
        (24, 32, 2, 1, 1)
    )
    
    enc_mlp_layers = (128, 64, 32) 
    
    latent_size = 24
    
    encoder_final_conv_shape = (32, 2, 2) 
    
    original_img_dims = (64, 64)

    vae = VAE(
        conv_layers_encoder_config=enc_conv_layers,
        mlp_layers_encoder_config=enc_mlp_layers,
        reparam_size=latent_size,
        encoder_conv_output_shape=encoder_final_conv_shape,
        original_image_dims=original_img_dims
    )

    summary(vae, (1, 64, 64))
