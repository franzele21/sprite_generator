"""
Better approaches for implementing skip connections in your encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        # Skip connection
        out += identity
        out = self.relu(out)
        
        return out

class SkipMLP(nn.Module):
    def __init__(self, input_size, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResNetBlock(input_size) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        all_output = []
        for block in self.blocks:
            x = block(x)
            all_output.append(x)
        return all_output

# Your Improved Encoder Class
class Encoder(nn.Module):
    def __init__(self, 
                 conv_layers: tuple[tuple[int]], 
                 mlp_layers: tuple[int], 
                 reparam_size: int,
                 num_resnet_blocks: int = 4,
                 randomize: bool = True,
                 return_mean_logvar: bool = False,
                 rand_intensity: float = 0.5
                 ):
        super().__init__()
        self.randomize = randomize
        self.return_mean_logvar = return_mean_logvar
        self.rand_intensity = rand_intensity

        # Build convolutional layers
        conv_modules = []
        for layer_params in conv_layers:
            conv_modules.extend([
                nn.Conv2d(*layer_params),
                nn.BatchNorm2d(layer_params[1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        self.conv = nn.Sequential(*conv_modules)

        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.randn(2, conv_layers[0][0], 64, 64)
            conv_output = self.conv(dummy_input)
            self.conv_output_size = conv_output[0]
    
        self.conversion = nn.Linear(self.conv_output_size.numel(), mlp_layers[0])

        # ResNet blocks with skip connections
        self.resnet_blocks = SkipMLP(mlp_layers[0], num_resnet_blocks)

        # Regular MLP layers
        mlp_modules = []
        prev_size = mlp_layers[0]
        
        for hidden_size in mlp_layers[1:]:
            mlp_modules.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2, inplace=True),
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
        
        # Apply ResNet blocks with skip connections
        z = self.conversion(z)
        resnet_output = self.resnet_blocks(z)
        z = resnet_output[-1]

        # Regular MLP
        z = self.mlp(z)

        # Reparameterization
        mean_val = self.mean_layer(z)
        log_var_val = self.logvar_layer(z)
        z = self.reparameterization(mean_val, log_var_val)

        if self.return_mean_logvar:
            return z, mean_val, log_var_val, resnet_output
        else:
            return z
    
    def reparameterization(self, mean, log_var):
        if self.randomize:
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std) * self.rand_intensity
            z = mean + std * epsilon
        else:
            z = mean
        return z


# Example usage and testing
if __name__ == "__main__":
    # Test the ResNet block
    block = ResNetBlock(128)
    x = torch.randn(32, 128)  # batch_size=32, features=128
    output = block(x)
    print(f"ResNet block input shape: {x.shape}")
    print(f"ResNet block output shape: {output.shape}")
    
    # Test the improved encoder
    conv_layers = (
        (1, 16, 16, 8, 0),
        (16, 24, 8, 4, 2),
        (24, 32, 2, 1, 1)
    )
    mlp_layers = (128, 64, 32)
    reparam_size = 24
    
    encoder = Encoder(
        conv_layers=conv_layers,
        mlp_layers=mlp_layers,
        reparam_size=reparam_size,
        num_resnet_blocks=4
    )
    
    # Test forward pass
    test_input = torch.randn(8, 1, 64, 64)
    output = encoder(test_input)
    print(f"Encoder input shape: {test_input.shape}")
    print(f"Encoder output shape: {output.shape}")