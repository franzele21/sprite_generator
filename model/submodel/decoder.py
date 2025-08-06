import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResNetBlock(nn.Module):
    """ResNet block for skip connections in MLP layers"""
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
    """MLP with ResNet-style skip connections"""
    def __init__(self, input_size, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResNetBlock(input_size) for _ in range(num_blocks)
        ])
        
    def forward(self, x, skip_connexions_input):
        if isinstance(skip_connexions_input, type(None)):
            skip_connexions_input = [torch.zeros_like(x) for i in range(len(self.blocks))]
        for block, input_ in zip(self.blocks, skip_connexions_input[::-1]):
            x = block(x+input_)
        return x


# class FunnelBlock(nn.Module):
#     """
#     Enhanced funnel block with pooling for feature integration
#     """
#     def __init__(self, in_channels: int, out_channels: int, scale_factor: tuple[float, float],
#                  pooling_type: str = "adaptive_avg", use_skip: bool = False):
#         super().__init__()
#         self.use_skip = use_skip
#         self.scale_factor = scale_factor
        
#         # Pooling layer for feature integration
#         if pooling_type == "max":
#             self.pool = nn.AdaptiveMaxPool2d((in_channels // 4, in_channels // 4))
#         elif pooling_type == "avg":
#             self.pool = nn.AdaptiveAvgPool2d((in_channels // 4, in_channels // 4))
#         elif pooling_type == "adaptive_avg":
#             self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
#         elif pooling_type == "adaptive_max":
#             self.pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
#         else:
#             self.pool = nn.Identity()
        
#         # Main convolution path
#         self.conv_path = nn.Sequential(
#             # First conv - feature refinement
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.LeakyReLU(0.2),
            
#             # Second conv - channel reduction
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2),
#         )
        
#         # Skip connection adaptation (if using skip connections)
#         if use_skip:
#             self.skip_adapt = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
#         # Upsampling
#         self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
#     def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
#         # Apply pooling for feature integration (helps reduce pixelation)
#         if not isinstance(self.pool, nn.Identity):
#             # Get pooled features and broadcast back
#             pooled = self.pool(x)
#             pooled_expanded = F.interpolate(pooled, size=x.shape[-2:], mode='bilinear', align_corners=False)
#             x = x + 0.1 * pooled_expanded  # Subtle feature integration
        
#         # Main convolution path
#         out = self.conv_path(x)
        
#         # Skip connection
#         if self.use_skip and skip is not None:
#             # Adapt skip connection to match current tensor
#             skip_adapted = self.skip_adapt(skip)
#             skip_resized = F.interpolate(skip_adapted, size=out.shape[-2:], mode='bilinear', align_corners=False)
#             out = out + 0.3 * skip_resized  # Weighted skip connection
        
#         # Upsample
#         out = self.upsample(out)
        
#         return out


class Decoder(nn.Module):    
    def __init__(self,
                 conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...],
                 reparam_size: int,
                 num_resnet_blocks: int,
                 expansion_factor: int,
                 encoder_conv_output_shape: tuple[int, int, int],
                 original_image_dims: tuple[int, int],
                ):
        super().__init__()
        self.encoder_conv_output_shape = encoder_conv_output_shape
        self.reparam_size = reparam_size
        self.original_image_dims = original_image_dims
        self.expansion_factor = expansion_factor

        # Calculate expanded conv shape
        C_orig, H_orig, W_orig = self.encoder_conv_output_shape
        self.expanded_conv_channels = C_orig * expansion_factor
        
        # Expand spatial dimensions for more detail
        self.expanded_conv_shape = (self.expanded_conv_channels, H_orig * 2, W_orig * 2)
        
        
        # --- 1. Enhanced MLP with ResNet blocks ---
        # Build MLP layers with significant expansion
        expanded_mlp_nodes = [
            self.reparam_size,
            *mlp_layers_encoder_config
        ]
        
        mlp_layers = []
        for i in range(len(expanded_mlp_nodes) - 1):
            mlp_layers.append(nn.Linear(expanded_mlp_nodes[i], expanded_mlp_nodes[i+1]))
            mlp_layers.append(nn.BatchNorm1d(expanded_mlp_nodes[i+1]))
            mlp_layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Add ResNet blocks for enhanced feature processing
        self.resnet_blocks = SkipMLP(mlp_layers_encoder_config[-1], num_resnet_blocks)
        self.conversion = nn.Linear(mlp_layers_encoder_config[-1], np.prod(self.expanded_conv_shape))
        
        self._build_transpose_conv_architecture(conv_layers_encoder_config)
    
    
    def _build_transpose_conv_architecture(self, conv_layers_encoder_config):
        """Build transpose convolution architecture with proper channel handling"""
        
        # First, we need to reduce from expanded channels to original encoder channels
        # Then build transpose convolutions
        
        # Initial channel reduction from expanded to original encoder output
        C_orig, H_orig, W_orig = self.encoder_conv_output_shape
        
        # Channel reduction layer
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(self.expanded_conv_channels, C_orig, kernel_size=3, padding=1),
            nn.BatchNorm2d(C_orig),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((H_orig, W_orig))  # Ensure correct spatial dimensions
        )
        
        # Build transpose convolutional layers (reverse of encoder)
        conv_layers = list(reversed(conv_layers_encoder_config))
        
        deconv_modules = []
        for i, layer_params in enumerate(conv_layers):
            if i < len(conv_layers) - 1:
                # Intermediate layers
                in_ch, out_ch = layer_params[1], layer_params[0]
                kernel_size = layer_params[2]
                stride = layer_params[3]
                padding = layer_params[4]
                
                deconv_modules.extend([
                    nn.Upsample(scale_factor=self.expansion_factor, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
                    nn.ConvTranspose2d(out_ch, out_ch, kernel_size, stride, padding),
                    nn.MaxPool2d(2, 2),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
            else:
                # Final layer - output image
                in_ch, out_ch = layer_params[1], layer_params[0]
                kernel_size = layer_params[2]
                stride = layer_params[3]
                padding = layer_params[4]
                
                deconv_modules.extend([
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
                    nn.Sigmoid()
                ])
        
        self.deconv = nn.Sequential(*deconv_modules)
        self.final_adjust = nn.AdaptiveAvgPool2d(self.original_image_dims)
    
    def forward(self, z: torch.Tensor, resnet_output=None) -> torch.Tensor:
        batch_size = z.shape[0]
        x = self.mlp(z)
        x = self.resnet_blocks(x, resnet_output)
        x = self.conversion(x)

        # 3. Reshape to high-dimensional conv tensor
        C_exp, H_exp, W_exp = self.expanded_conv_shape
        x = x.view(batch_size, C_exp, H_exp, W_exp)
        
        # Transpose convolution path
        # First reduce channels and spatial dimensions to match original encoder output
        x = self.channel_reduction(x)
        
        # Then apply transpose convolutions
        x = self.deconv(x)
        
        if x.shape[-2:] != self.original_image_dims:
            x = self.final_adjust(x)
        
        return x



if __name__ == "__main__":
    from torchsummary import summary
    # Test configuration
    enc_conv_layers = (
        (1, 16, 16, 8, 0),    
        (16, 24, 8, 4, 2),   
        (24, 32, 2, 1, 1)    
    )
    
    enc_mlp_layers = (128, 64, 32) 
    latent_size = 24
    encoder_final_conv_shape = (32, 2, 2) 
    original_img_dims = (64, 64)
    batch_size = 4
    
    # Test combined decoder with transpose conv architecture
    transpose_decoder = Decoder(
        conv_layers_encoder_config=enc_conv_layers,
        mlp_layers_encoder_config=enc_mlp_layers,
        reparam_size=latent_size,
        num_resnet_blocks=4,
        expansion_factor=8,
        encoder_conv_output_shape=encoder_final_conv_shape,
        original_image_dims=original_img_dims,
    )
    
    summary(transpose_decoder, (latent_size, ))