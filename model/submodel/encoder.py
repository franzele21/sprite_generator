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
                 skip_mlp_nb: int,
                 randomize: bool = True,
                 return_mean_logvar: bool = False,
                 rand_intensity: float = 0.5
                 ):
        assert skip_mlp_nb%2==0, f"skip_mlp_nb should be an even number ({skip_mlp_nb} is not)"
        super().__init__()
        self.eval()
        self.randomize = randomize
        self.return_mean_logvar = return_mean_logvar
        self.rand_intensity = rand_intensity

        # Build convolutional layers properly
        conv_modules = []
        for layer_params in conv_layers:
            conv_modules.extend([
                nn.Conv2d(*layer_params),
                nn.BatchNorm2d(layer_params[1]),
                nn.ReLU(inplace=True),
            ])
        self.conv = nn.Sequential(*conv_modules)

        # Calculate conv output size for first MLP layer
        with torch.no_grad():
            dummy_input = torch.randn(2, conv_layers[0][0], 64, 64)
            conv_output = self.conv(dummy_input)
            self.conv_output_size = conv_output[0].numel()

        prev_size = self.conv_output_size
        
        resnet_modules = []
        for i in range(skip_mlp_nb):
            resnet_modules.extend(
                [nn.Linear(prev_size, prev_size),
                 nn.BatchNorm1d(prev_size),
                 nn.ReLU(inplace=True)]
            )
        self.resnet = nn.Sequential(*resnet_modules)

        # Build MLP layers - first layer connects to flattened conv output
        mlp_modules = []
        
        for hidden_size in mlp_layers:
            mlp_modules.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        self.mlp = nn.Sequential(*mlp_modules)

        # Reparameterization layers
        self.mean_layer = nn.Linear(mlp_layers[-1], reparam_size)
        self.logvar_layer = nn.Linear(mlp_layers[-1], reparam_size)
    
    def forward(self, x):
        # Convolutional feature extraction
        z = self.conv(x)
        z = torch.flatten(z, start_dim=1)
        skip_connexion = []
        for i, layer in enumerate(self.resnet):
            match type(layer):
                case nn.modules.linear.Linear:
                    if i > 5:
                        z = layer(z+skip_connexion[-2])
                case nn.modules.batchnorm.BatchNorm1d:
                    z = layer(z)
                case nn.modules.activation.ReLU:
                    z = layer(z)
                    skip_connexion.append(z)
        z = self.mlp(z)

        # Get mean and log variance for reparameterization
        mean_val = self.mean_layer(z)
        log_var_val = self.logvar_layer(z)

        # Apply reparameterization trick
        z = self.reparameterization(mean_val, log_var_val)

        if self.return_mean_logvar:
            return z, mean_val, log_var_val
        else:
            return z
    
    def reparameterization(self, mean, log_var):
        """
        Proper reparameterization trick implementation:
        z = μ + σ * ε, where σ = exp(0.5 * log_var) and ε ~ N(0,1)
        
        Args:
            mean: Mean of the latent distribution
            log_var: Log variance of the latent distribution (not variance!)
        """
        if self.randomize:
            # Standard deviation from log variance
            std = torch.exp(0.5 * log_var)
            # Sample epsilon from standard normal distribution
            epsilon = torch.randn_like(std) * self.rand_intensity
            # Reparameterization: z = μ + σ * ε
            z = mean + std * epsilon
        else:
            # Return mean without randomization
            z = mean
        return z

if __name__ == "__main__":
    from torchsummary import summary

    conv_layers = (
        (1, 16, 16, 8, 0),    # Input: (1, 64, 64) -> Output: (16, 7, 7)
        (16, 24, 8, 4, 2),    # Input: (16, 7, 7) -> Output: (24, 1, 1) 
        (24, 32, 2, 1, 1)     # Input: (24, 1, 1) -> Output: (32, 2, 2)
    )
    mlp_layers = (128, 64, 32)  # 128 -> 64 -> 32
    reparam_size = 24
    skip_mlp_nb = 8

    # Fix device selection logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = Encoder(conv_layers, mlp_layers, skip_mlp_nb, reparam_size, return_mean_logvar=True)
    
    # Test the encoder
    print("Encoder Model Structure:")
    summary(encoder, (1, 64, 64), device=str(device))