import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedDecoder(nn.Module):
    """
    Enhanced Decoder with Funnel Architecture for R-Conv-VAE
    --------------------------------------------------------
    This decoder uses a funnel-like structure to produce clearer reconstructions:
    1. Significantly increases dimensionality in early stages
    2. Uses progressive refinement with pooling operations
    3. Implements skip connections for better gradient flow
    4. Uses batch normalization for training stability
    
    The architecture follows: MLP -> Upsample + High Dim Conv -> Funnel Conv + Pooling -> Final Refinement
    """
    
    def __init__(self,
                 conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...],
                 reparam_size: int,
                 encoder_conv_output_shape: tuple[int, int, int],
                 original_image_dims: tuple[int, int],
                 expansion_factor: int = 8,  # How much to expand dimensionality initially
                 use_skip_connections: bool = True,
                 pooling_type: str = "adaptive_avg"  # "max", "avg", "adaptive_avg", "adaptive_max"
                ):
        super().__init__()
        self.encoder_conv_output_shape = encoder_conv_output_shape
        self.reparam_size = reparam_size
        self.original_image_dims = original_image_dims
        self.expansion_factor = expansion_factor
        self.use_skip_connections = use_skip_connections
        self.pooling_type = pooling_type
        
        # Calculate expanded conv shape first
        C_orig, H_orig, W_orig = self.encoder_conv_output_shape
        self.expanded_conv_channels = C_orig * expansion_factor
        
        # We'll upsample the spatial dimensions as well for more detail
        self.expanded_conv_shape = (self.expanded_conv_channels, H_orig * 2, W_orig * 2)
        
        # Calculate the required MLP output size to match the expanded conv input
        self.expanded_mlp_size = (self.expanded_conv_shape[0] * 
                                 self.expanded_conv_shape[1] * 
                                 self.expanded_conv_shape[2])
        
        # --- 1. Enhanced MLP Part with Significant Expansion ---
        # First expand to much higher dimensionality, then gradually reduce
        expanded_mlp_nodes = [
            self.reparam_size,
            self.reparam_size * 4,      # 4x expansion
            self.reparam_size * 8,      # 8x expansion  
            self.expanded_mlp_size      # Final expanded size (calculated above)
        ]
        
        mlp_layers = []
        for i in range(len(expanded_mlp_nodes) - 1):
            mlp_layers.append(nn.Linear(expanded_mlp_nodes[i], expanded_mlp_nodes[i+1]))
            mlp_layers.append(nn.BatchNorm1d(expanded_mlp_nodes[i+1]))
            mlp_layers.append(nn.LeakyReLU(0.2))
            mlp_layers.append(nn.Dropout(0.1))  # Light regularization
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # --- 2. Initial Upsampling and High-Dimensional Convolutions ---
        # Start with very high channel count and gradually reduce
        
        initial_conv_layers = []
        
        # First conv block - refine the high-dimensional features
        initial_conv_layers.extend([
            nn.Conv2d(self.expanded_conv_channels, self.expanded_conv_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(self.expanded_conv_channels),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(self.expanded_conv_channels, self.expanded_conv_channels // 2, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(self.expanded_conv_channels // 2),
            nn.LeakyReLU(0.2),
        ])
        
        self.initial_conv = nn.Sequential(*initial_conv_layers)
        
        # --- 3. Funnel Architecture with Progressive Refinement ---
        self.funnel_blocks = nn.ModuleList()
        
        # Calculate progression from current size to target size
        current_channels = self.expanded_conv_channels // 2
        current_h, current_w = self.expanded_conv_shape[1], self.expanded_conv_shape[2]
        target_h, target_w = self.original_image_dims
        target_channels = conv_layers_encoder_config[0][0]  # Final output channels
        
        # Create funnel blocks that progressively:
        # 1. Increase spatial resolution
        # 2. Decrease channel count
        # 3. Apply pooling for feature integration
        
        num_funnel_blocks = 4
        channel_reduction_factor = (current_channels / target_channels) ** (1 / num_funnel_blocks)
        
        for i in range(num_funnel_blocks):
            next_channels = max(target_channels, int(current_channels / channel_reduction_factor))
            
            # Calculate target spatial size for this block
            scale_factor = ((target_h / current_h) ** (1 / (num_funnel_blocks - i)), 
                           (target_w / current_w) ** (1 / (num_funnel_blocks - i)))
            
            block = FunnelBlock(
                in_channels=current_channels,
                out_channels=next_channels,
                scale_factor=scale_factor,
                pooling_type=self.pooling_type,
                use_skip=self.use_skip_connections and i > 0
            )
            
            self.funnel_blocks.append(block)
            current_channels = next_channels
            current_h = int(current_h * scale_factor[0])
            current_w = int(current_w * scale_factor[1])
        
        # --- 4. Final Refinement Layers ---
        self.final_refinement = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_channels),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(current_channels, target_channels, kernel_size=1),  # 1x1 conv for channel adjustment
            nn.Sigmoid()  # Output activation
        )
        
        # --- 5. Final Upsampling (if needed) ---
        self.final_upsample = None
        if current_h != target_h or current_w != target_w:
            self.final_upsample = nn.Upsample(size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        
        # 1. MLP expansion
        x = self.mlp(z)
        
        # 2. Reshape to high-dimensional conv tensor
        C_exp, H_exp, W_exp = self.expanded_conv_shape
        x = x.view(batch_size, C_exp, H_exp, W_exp)
        
        # 3. Initial convolution refinement
        x = self.initial_conv(x)
        
        # 4. Progressive funnel refinement
        skip_connections = []
        for i, block in enumerate(self.funnel_blocks):
            if self.use_skip_connections and i > 0:
                x = block(x, skip_connections[-1] if skip_connections else None)
            else:
                x = block(x)
            
            # Store for potential skip connections
            if self.use_skip_connections:
                skip_connections.append(x.clone())
        
        # 5. Final refinement
        x = self.final_refinement(x)
        
        # 6. Final upsampling if needed
        if self.final_upsample is not None:
            x = self.final_upsample(x)
        
        return x


class FunnelBlock(nn.Module):
    """
    A single block in the funnel architecture that:
    1. Applies pooling for feature integration
    2. Upsamples spatially
    3. Reduces channels
    4. Optionally uses skip connections
    """
    
    def __init__(self, in_channels: int, out_channels: int, scale_factor: tuple[float, float],
                 pooling_type: str = "adaptive_avg", use_skip: bool = False):
        super().__init__()
        self.use_skip = use_skip
        self.scale_factor = scale_factor
        
        # Pooling layer for feature integration
        if pooling_type == "max":
            self.pool = nn.AdaptiveMaxPool2d((in_channels // 4, in_channels // 4))
        elif pooling_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d((in_channels // 4, in_channels // 4))
        elif pooling_type == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        elif pooling_type == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        else:
            self.pool = nn.Identity()
        
        # Main convolution path
        self.conv_path = nn.Sequential(
            # First conv - feature refinement
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            
            # Second conv - channel reduction
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        
        # Skip connection adaptation (if using skip connections)
        if use_skip:
            self.skip_adapt = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        # Apply pooling for feature integration (helps reduce pixelation)
        if not isinstance(self.pool, nn.Identity):
            # Get pooled features and broadcast back
            pooled = self.pool(x)
            pooled_expanded = F.interpolate(pooled, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x + 0.1 * pooled_expanded  # Subtle feature integration
        
        # Main convolution path
        out = self.conv_path(x)
        
        # Skip connection
        if self.use_skip and skip is not None:
            # Adapt skip connection to match current tensor
            skip_adapted = self.skip_adapt(skip)
            skip_resized = F.interpolate(skip_adapted, size=out.shape[-2:], mode='bilinear', align_corners=False)
            out = out + 0.3 * skip_resized  # Weighted skip connection
        
        # Upsample
        out = self.upsample(out)
        
        return out


# Legacy wrapper to maintain compatibility with existing code
class Decoder(EnhancedDecoder):
    """Wrapper class to maintain backward compatibility"""
    def __init__(self, *args, **kwargs):
        # Set default enhanced parameters
        enhanced_kwargs = {
            'expansion_factor': 8,
            'use_skip_connections': True,
            'pooling_type': 'adaptive_avg'
        }
        enhanced_kwargs.update(kwargs)
        super().__init__(*args, **enhanced_kwargs)


if __name__ == "__main__":
    # --- Configuration ---
    enc_conv_layers = (
        (1, 16, 16, 8, 0),    
        (16, 24, 8, 4, 2),   
        (24, 32, 2, 1, 1)    
    )
    
    enc_mlp_layers = (128, 64, 32) 
    latent_size = 24
    encoder_final_conv_shape = (32, 2, 2) 
    original_img_dims = (64, 64)

    # --- Test Enhanced Decoder ---
    print("Testing Enhanced Decoder...")
    enhanced_decoder = EnhancedDecoder(
        conv_layers_encoder_config=enc_conv_layers,
        mlp_layers_encoder_config=enc_mlp_layers,
        reparam_size=latent_size,
        encoder_conv_output_shape=encoder_final_conv_shape,
        original_image_dims=original_img_dims,
        expansion_factor=8,
        use_skip_connections=True,
        pooling_type="adaptive_avg"
    )

    batch_size = 4
    dummy_latent_z = torch.randn(batch_size, latent_size)
    reconstructed_image = enhanced_decoder(dummy_latent_z)

    print(f"Enhanced Decoder instantiated successfully.")
    print(f"Input latent z shape: {dummy_latent_z.shape}")
    print(f"Reconstructed image shape: {reconstructed_image.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in enhanced_decoder.parameters()):,}")

    # Print model summary
    try:
        from torchsummary import summary
        print("\nEnhanced Decoder Summary:")
        summary(enhanced_decoder, input_size=(latent_size,))
    except ImportError:
        print("\ntorchsummary not installed. Skipping summary.")
    except Exception as e:
        print(f"\nError during summary: {e}")

    # Test backward compatibility
    print("\nTesting backward compatibility...")
    legacy_decoder = Decoder(
        conv_layers_encoder_config=enc_conv_layers,
        mlp_layers_encoder_config=enc_mlp_layers,
        reparam_size=latent_size,
        encoder_conv_output_shape=encoder_final_conv_shape,
        original_image_dims=original_img_dims
    )
    
    legacy_output = legacy_decoder(dummy_latent_z)
    print(f"Legacy wrapper output shape: {legacy_output.shape}")
    
    # Compare parameter counts
    print(f"Enhanced decoder parameters: {sum(p.numel() for p in enhanced_decoder.parameters()):,}")
    print(f"Legacy wrapper parameters: {sum(p.numel() for p in legacy_decoder.parameters()):,}")
    
    # Print legacy summary
    try:
        print("\nLegacy Decoder Summary:")
        summary(legacy_decoder, input_size=(latent_size,))
    except ImportError:
        print("\ntorchsummary not installed. Skipping summary.")
    except Exception as e:
        print(f"\nError during legacy summary: {e}")