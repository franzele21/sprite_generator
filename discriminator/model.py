"""
Possible Discriminator of this project.
Could be used to enhance the performance of the generator.
Is a classifier, that will say if the input is a Pokémon or not.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, 
                 conv_layers: tuple[tuple[int]], 
                 mlp_layers: tuple[int], 
                 ):
        super().__init__()

        self.conv = nn.Sequential(
            *[[nn.Conv2d(*conv_layers[i//2]), nn.Sigmoid()][i%2] for i in range(len(conv_layers)*2)]
        )
        self.mlp = nn.Sequential(
            *[[nn.Linear(mlp_layers[i//2], mlp_layers[i//2+1]), nn.ReLU()][i%2] for i in range(len(mlp_layers)*2-2)]
        )
        self.last_layer = nn.Sequential(
            (nn.Linear(mlp_layers[-1], 1),
             nn.Sigmoid())
        )
    
    def forward(self, x):
        z = self.conv(x)
        z = self.mlp(z)
        y = self.last_layer(z)

        return y
    
if __name__ == "__main__":
    from torchsummary import summary

    conv_layers = (
        (1, 16, 16, 8, 0),
        (16, 24, 8, 4, 2),
        (24, 32, 2, 1, 1)
    )
    mlp_layers = (128, 64, 32)

    discri = Discriminator(conv_layers, mlp_layers)
    summary(discri, (1, 64, 64), device="cpu")