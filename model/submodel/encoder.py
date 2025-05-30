"""
Encoder of the R-Conv-VAE
Should take an image as input, and produce a latent space.
The reparameterisation trick will be implemented here too.

Should be less complex than the Decoder, because it has less work to do.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder
    -------
    Args:
        conv_layers (tuple[tuple[int]]) : Architecture of the initial convolution layers. Should have at least one entry. 
        Every index is for an argument of nn.Conv2d (in place).
        So (((1, 16, 16, 8, 1))) -> nn.Conv2d(1, 16, 16, 8, 1) -> Conv2d(1, 16, kernel_size=(16, 16), stride=(8, 8), padding=(1, 1))

        mlp_layers (tuple[int]) : Architecture of the MultiLayerPerception layers. Should have at least two entry.
        Every index is the size of the layer. The last index won't produce a layer.
        So (128, 64, 32) -> nn.Linear(128, 64) + nn.Linear(64, 32) ->  Linear(in_features=128, out_features=64, bias=True) + Linear(in_features=64, out_features=32, bias=True)

        reparam_size (int) : Size of the output of the reparameterization layers (`log_var` and `mean`)

        randomize (bool) : If False, won't randomize in the reparameterization trick.

        return_mean_logvar (bool) : If True, will return (z, mean_val, log_var_val)
    
    """
    def __init__(self, 
                 conv_layers: tuple[tuple[int]], 
                 mlp_layers: tuple[int], 
                 reparam_size: int,
                 randomize: bool=True,
                 return_mean_logvar: bool=False,
                 rand_intensity: float=0.5
                 ):
        super().__init__()
        self.randomize = randomize
        self.return_mean_logvar = return_mean_logvar
        self.rand_intensity = rand_intensity

        self.conv = nn.Sequential(
            *[[nn.Conv2d(*conv_layers[i//3]), 
            nn.ReLU(),
            nn.BatchNorm2d(conv_layers[i//3][1])][i%3] for i in range(len(conv_layers)*3)]
        )
        self.mlp = nn.Sequential(
            *[[nn.Linear(mlp_layers[i//2], mlp_layers[i//2+1]), nn.ReLU()][i%2] for i in range(len(mlp_layers)*2-2)]
        )

        self.log_var = nn.Linear(mlp_layers[-1], reparam_size)
        self.mean = nn.Linear(mlp_layers[-1], reparam_size)
    
    def forward(self, x):
        z = self.conv(x)
        z = torch.flatten(z, start_dim=1)
        z = self.mlp(z)

        mean_val, log_var_val = self.mean(z), self.log_var(z)

        z = self.reparameterization(mean_val, log_var_val)

        if self.return_mean_logvar:
            return z, mean_val, log_var_val
        else:
            return z
    
    def reparameterization(self, mean, var):
        if self.randomize:
            epsilon = torch.randn_like(var) * self.rand_intensity
        else:
            epsilon = torch.ones_like(var)
        z = mean + var*epsilon 
        return z


if __name__ == "__main__":
    from torchsummary import summary

    conv_layers = (
        (1, 16, 16, 8, 0),
        (16, 24, 8, 4, 2),
        (24, 32, 2, 1, 1)
    )
    mlp_layers = (128, 64, 32)
    reparam_size = 24

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(device)

    encoder = Encoder(conv_layers, mlp_layers, reparam_size)
    summary(encoder, (1, 64, 64), device="cpu")

