"""
Encoder of the R-Conv-VAE
Should take an image as input, and produce a latent space.
The reparameterisation trick will be implemented here too.

Should be less complex than the Decoder, because it has less work to do.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SubEncoder(nn.Module):
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
                 randomize: bool = True,
                 return_mean_logvar: bool = False):
        super().__init__()
        self.randomize = randomize
        self.return_mean_logvar = return_mean_logvar

        # Build convolutional layers properly
        conv_modules = []
        for layer_params in conv_layers:
            conv_modules.extend([
                nn.Conv2d(*layer_params),
                nn.ReLU(inplace=True),  # Changed from Sigmoid to ReLU
                nn.BatchNorm2d(layer_params[1])
            ])
        self.conv = nn.Sequential(*conv_modules)
        
        # Build MLP layers properly
        mlp_modules = []
        for i in range(len(mlp_layers) - 1):
            mlp_modules.extend([
                nn.Linear(mlp_layers[i], mlp_layers[i + 1]),
                nn.ReLU(inplace=True)
            ])
        self.mlp = nn.Sequential(*mlp_modules)

        # Reparameterization layers
        self.mean_layer = nn.Linear(mlp_layers[-1], reparam_size)
        self.logvar_layer = nn.Linear(mlp_layers[-1], reparam_size)
    
    def forward(self, x):
        # Convolutional feature extraction
        z = self.conv(x)
        z = torch.flatten(z, start_dim=1)
        z = self.mlp(z)

        # Get mean and log variance
        mean = self.mean_layer(z)
        logvar = self.logvar_layer(z)

        # Reparameterization trick
        z_sampled = self.reparameterization(mean, logvar)

        if self.return_mean_logvar:
            return z_sampled, mean, logvar
        else:
            return z_sampled
    
    def reparameterization(self, mean, logvar):
        """
        Proper reparameterization trick: z = μ + σ * ε
        where σ = exp(0.5 * log_var) and ε ~ N(0,1)
        """
        if self.randomize:
            # Standard deviation from log variance
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            return mean

class Encoder(nn.Module):
    def __init__(self,
                 n_encoders: int,
                 conv_layers: tuple[tuple[int]], 
                 mlp_layers: tuple[int], 
                 reparam_size: int,
                 randomize: bool = True,
                 return_mean_logvar: bool = True):
        super().__init__()
        self.return_mean_logvar = return_mean_logvar
        self.n_encoders = n_encoders
        self.reparam_size = reparam_size

        # Use ModuleList for proper parameter registration
        self.encoders = nn.ModuleList([
            SubEncoder(conv_layers, mlp_layers, reparam_size, 
                      randomize, return_mean_logvar)
            for _ in range(n_encoders)
        ])
        
        # Embedding layer to combine multiple encoder outputs
        self.embedding = nn.Linear(reparam_size * n_encoders, reparam_size)
    
    def forward(self, x):
        encoder_outputs = [encoder(x) for encoder in self.encoders]

        if self.return_mean_logvar:
            # Separate z, mean, and logvar
            all_z = torch.cat([output[0] for output in encoder_outputs], dim=1)
            all_mean = torch.cat([output[1] for output in encoder_outputs], dim=1)
            all_logvar = torch.cat([output[2] for output in encoder_outputs], dim=1)
            
            # Combine representations
            z = F.relu(self.embedding(all_z))
            
            # Average the mean and logvar (or you could embed them too)
            combined_mean = torch.mean(all_mean.view(-1, self.n_encoders, self.reparam_size), dim=1)
            combined_logvar = torch.mean(all_logvar.view(-1, self.n_encoders, self.reparam_size), dim=1)
            
            return z, combined_mean, combined_logvar
        else:
            all_z = torch.cat([output for output in encoder_outputs], dim=1)
            z = F.relu(self.embedding(all_z))
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

    encoder = Encoder(5, conv_layers, mlp_layers, reparam_size)
    summary(encoder, (1, 64, 64), device="cpu")

