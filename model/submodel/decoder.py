import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Decoder of the R-Conv-VAE
    --------------------------
    Takes a latent space vector `z` as input and reconstructs an image.
    It is designed to be the inverse of the provided Encoder structure.

    Args:
        conv_layers_encoder_config (tuple[tuple[int, ...], ...]):
            The architecture of the Encoder's convolutional layers.
            Each inner tuple contains arguments for `nn.Conv2d`
            (e.g., (in_channels, out_channels, kernel_size, stride, padding)).

        mlp_layers_encoder_config (tuple[int, ...]):
            The architecture of the Encoder's MLP layers (sizes of layers).
            Example: (128, 64, 32) means Linear(128,64) then Linear(64,32).
            The first element is the flattened size of the Encoder's conv output.

        reparam_size (int):
            Size of the latent space vector `z` (input to this Decoder).

        encoder_conv_output_shape (tuple[int, int, int]):
            The shape (Channels, Height, Width) of the output tensor from the
            Encoder's final convolutional layer (before flattening).

        original_image_dims (tuple[int, int]):
            The spatial dimensions (Height, Width) of the original image
            that was fed into the Encoder. This is crucial for calculating
            the correct `output_padding` for the transposed convolutions
            to reconstruct the image to its original size.
    """
    def __init__(self,
                 conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...],
                 reparam_size: int,
                 encoder_conv_output_shape: tuple[int, int, int], # (Channels, Height, Width)
                 original_image_dims: tuple[int, int] # (Height, Width)
                ):
        super().__init__()
        self.encoder_conv_output_shape = encoder_conv_output_shape
        self.reparam_size = reparam_size
        self.original_image_dims = original_image_dims

        # --- 1. MLP Part ---
        # Reverses the Encoder's MLP structure.
        # Encoder MLP went from mlp_layers_encoder_config[0] down to mlp_layers_encoder_config[-1].
        # Decoder MLP goes from reparam_size up to mlp_layers_encoder_config[0].
        
        # Node sizes for the Decoder's MLP layers
        # Example: mlp_layers_encoder_config = (128, 64, 32), reparam_size = 24
        # decoder_mlp_nodes will be [24, 32, 64, 128] (from latent to flattened conv size)
        decoder_mlp_nodes = [self.reparam_size] + list(mlp_layers_encoder_config)[::-1]
        
        mlp_decoder_layers = []
        for i in range(len(decoder_mlp_nodes) - 1):
            mlp_decoder_layers.append(nn.Linear(decoder_mlp_nodes[i], decoder_mlp_nodes[i+1]))
            mlp_decoder_layers.append(nn.ReLU()) # Matching Encoder's MLP activation (ReLU)
        
        self.mlp = nn.Sequential(*mlp_decoder_layers)
        
        # Output of MLP has size decoder_mlp_nodes[-1], which is mlp_layers_encoder_config[0].
        # This must be equal to C_encoder_conv_out * H_encoder_conv_out * W_encoder_conv_out.
        expected_mlp_output_dim = mlp_layers_encoder_config[0]
        actual_mlp_output_dim = (self.encoder_conv_output_shape[0] *
                                 self.encoder_conv_output_shape[1] *
                                 self.encoder_conv_output_shape[2])
        if expected_mlp_output_dim != actual_mlp_output_dim:
            raise ValueError(
                f"Decoder MLP output size mismatch: Expected {expected_mlp_output_dim} (from mlp_layers_encoder_config[0]), "
                f"but got {actual_mlp_output_dim} (from flattening encoder_conv_output_shape {self.encoder_conv_output_shape}). "
                f"Ensure configurations are consistent."
            )

        # --- 2. Transposed Convolutional Part ---
        # This part reconstructs the image from the reshaped MLP output.
        # It requires calculating intermediate shapes from the encoder to determine
        # the correct `output_padding` for each `ConvTranspose2d` layer.

        # Store details of each encoder convolutional layer's spatial transformations
        self.encoder_conv_details = [] 
        
        H_curr, W_curr = self.original_image_dims
        for i in range(len(conv_layers_encoder_config)):
            enc_cfg = conv_layers_encoder_config[i]
            # Assuming enc_cfg format: (in_channels, out_channels, kernel_size, stride, padding, ...)
            K_orig, S_orig, P_orig = enc_cfg[2], enc_cfg[3], enc_cfg[4]

            K_h, K_w = self._get_hw_params(K_orig, "kernel_size")
            S_h, S_w = self._get_hw_params(S_orig, "stride")
            P_h, P_w = self._get_hw_params(P_orig, "padding")

            H_in_enc, W_in_enc = H_curr, W_curr # Input dimensions to this encoder conv layer
            # Calculate output dimensions of this encoder conv layer
            H_out_enc = (H_in_enc + 2 * P_h - K_h) // S_h + 1
            W_out_enc = (W_in_enc + 2 * P_w - K_w) // S_w + 1
            
            self.encoder_conv_details.append({
                'H_in_enc': H_in_enc, 'W_in_enc': W_in_enc, 
                'H_out_enc': H_out_enc, 'W_out_enc': W_out_enc,
                'K_h': K_h, 'K_w': K_w, 'S_h': S_h, 'S_w': S_w, 
                'P_h': P_h, 'P_w': P_w,
                'enc_in_c': enc_cfg[0], 'enc_out_c': enc_cfg[1]
            })
            H_curr, W_curr = H_out_enc, W_out_enc # Output becomes input for the next encoder layer

        # Verify that the final calculated H,W match the provided encoder_conv_output_shape's spatial part
        if not (H_curr == self.encoder_conv_output_shape[1] and W_curr == self.encoder_conv_output_shape[2]):
            raise ValueError(
                f"Calculated final encoder spatial dimensions ({H_curr}, {W_curr}) mismatch "
                f"the provided encoder_conv_output_shape's spatial dimensions "
                f"({self.encoder_conv_output_shape[1]}, {self.encoder_conv_output_shape[2]}). "
                f"Please check original_image_dims and conv_layers_encoder_config."
            )

        # Build the transposed convolutional layers for the decoder
        deconv_layers = []
        num_encoder_conv_layers = len(conv_layers_encoder_config)

        # Initial H, W for deconvolution process is the output of the last encoder convolution
        H_in_deconv, W_in_deconv = self.encoder_conv_output_shape[1], self.encoder_conv_output_shape[2]

        for i in range(num_encoder_conv_layers):
            # Deconv layers are built in reverse order of encoder conv layers
            enc_conv_idx = num_encoder_conv_layers - 1 - i
            details = self.encoder_conv_details[enc_conv_idx]

            # Deconv layer parameters (mirrored from encoder's conv layer)
            dec_in_c = details['enc_out_c']  # Input channels for deconv = output channels of corresponding enc_conv
            dec_out_c = details['enc_in_c'] # Output channels for deconv = input channels of corresponding enc_conv
            
            K_h, K_w = details['K_h'], details['K_w']
            S_h, S_w = details['S_h'], details['S_w']
            P_h, P_w = details['P_h'], details['P_w']

            # Target output dimensions for this deconv layer are the INPUT dimensions of the corresponding encoder conv layer
            H_out_target_deconv, W_out_target_deconv = details['H_in_enc'], details['W_in_enc']

            # Calculate output_padding for ConvTranspose2d.
            # Formula for ConvTranspose2d output: H_out = (H_in - 1)*S - 2*P + K + OP
            # So, OP = H_out_target - ((H_in_deconv - 1)*S - 2*P + K)
            op_h = H_out_target_deconv - ((H_in_deconv - 1) * S_h - 2 * P_h + K_h)
            op_w = W_out_target_deconv - ((W_in_deconv - 1) * S_w - 2 * P_w + K_w)

            # PyTorch constraint: 0 <= output_padding < stride
            if not (0 <= op_h < S_h if S_h > 0 else op_h == 0):
                # This might indicate an architecture that isn't perfectly reversible with standard output_padding,
                # or an edge case. Clamping is a pragmatic approach.
                print(f"Warning: Calculated op_h ({op_h}) for S_h={S_h} is out of range [0, S_h-1) for deconv layer {i}. Clamping.")
                op_h = max(0, min(op_h, S_h - 1 if S_h > 0 else 0)) # Clamp op_h
            if not (0 <= op_w < S_w if S_w > 0 else op_w == 0):
                print(f"Warning: Calculated op_w ({op_w}) for S_w={S_w} is out of range [0, S_w-1) for deconv layer {i}. Clamping.")
                op_w = max(0, min(op_w, S_w - 1 if S_w > 0 else 0)) # Clamp op_w
            
            deconv_layers.append(nn.ConvTranspose2d(
                in_channels=dec_in_c, 
                out_channels=dec_out_c, 
                kernel_size=(K_h, K_w), 
                stride=(S_h, S_w), 
                padding=(P_h, P_w),
                output_padding=(op_h, op_w)
            ))

            # Update H_in_deconv, W_in_deconv for the next deconv layer:
            # The input to the next deconv layer is the target output of this current deconv layer.
            H_in_deconv = H_out_target_deconv 
            W_in_deconv = W_out_target_deconv

            # Activation function:
            # Encoder used Sigmoid after each Conv. For Decoder:
            # - Intermediate layers: ReLU (common practice)
            # - Final layer: Sigmoid (to output image pixels in [0,1] range)
            if i < num_encoder_conv_layers - 1:
                deconv_layers.append(nn.ReLU()) 
            else:
                deconv_layers.append(nn.Sigmoid()) 
        
        self.deconv = nn.Sequential(*deconv_layers)

    @staticmethod
    def _get_hw_params(param_val, name_for_error_msg: str) -> tuple[int, int]:
        """Converts an int or tuple (K) or (K,K) to a (K_h, K_w) tuple."""
        if isinstance(param_val, int):
            return param_val, param_val
        elif isinstance(param_val, tuple) and len(param_val) == 1: # e.g., kernel_size=(k,)
            return param_val[0], param_val[0]
        elif isinstance(param_val, tuple) and len(param_val) == 2: # e.g., kernel_size=(kh, kw)
            return param_val[0], param_val[1]
        else:
            raise ValueError(
                f"{name_for_error_msg} must be an int or a tuple of 1 or 2 ints. Got {param_val}"
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder.
        Args:
            z (torch.Tensor): Latent space tensor of shape (batch_size, reparam_size).
        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        # 1. Pass latent vector through MLP
        x = self.mlp(z) # Output shape: (batch_size, mlp_layers_encoder_config[0])
        
        # 2. Reshape MLP output to 4D tensor for deconvolution
        # (batch_size, Channels_enc_conv_out, Height_enc_conv_out, Width_enc_conv_out)
        batch_size = z.shape[0]
        C_enc_out, H_enc_out, W_enc_out = self.encoder_conv_output_shape
        try:
            x = x.view(batch_size, C_enc_out, H_enc_out, W_enc_out)
        except RuntimeError as e:
            raise RuntimeError(
                f"Error reshaping MLP output in Decoder: {e}. "
                f"MLP output size: {x.shape}, Target view shape: ({batch_size}, {C_enc_out}, {H_enc_out}, {W_enc_out}). "
                f"Ensure mlp_layers_encoder_config[0] == C*H*W of encoder_conv_output_shape."
            ) from e

        # 3. Pass through transposed convolutional layers to reconstruct image
        x = self.deconv(x)
        
        return x


if __name__ == "__main__":
    from torchsummary import summary

    # --- Configuration mirroring the Encoder example ---
    # Encoder's convolutional layer configurations
    # (in_channels, out_channels, kernel_size, stride, padding)
    enc_conv_layers = (
        (1, 16, 16, 8, 0),    # Output: (16, 7, 7) for (1, 64, 64) input
        (16, 24, 8, 4, 2),   # Output: (24, 1, 1) for (16, 7, 7) input
        (24, 32, 2, 1, 1)    # Output: (32, 2, 2) for (24, 1, 1) input
    )
    
    # Encoder's MLP layer configurations (sizes of layers)
    # Input to MLP is flattened output of last conv: 32*2*2 = 128
    enc_mlp_layers = (128, 64, 32) 
    
    # Size of the latent space (reparameterization output)
    latent_size = 24
    
    # Shape of the output from Encoder's convolutional part (before flattening)
    # (Channels, Height, Width)
    encoder_final_conv_shape = (32, 2, 2) 
    
    # Original image dimensions fed to the Encoder
    original_img_dims = (64, 64) # (Height, Width)

    # --- Instantiate Decoder ---
    decoder = Decoder(
        conv_layers_encoder_config=enc_conv_layers,
        mlp_layers_encoder_config=enc_mlp_layers,
        reparam_size=latent_size,
        encoder_conv_output_shape=encoder_final_conv_shape,
        original_image_dims=original_img_dims
    )

    # --- Test with a dummy latent vector ---
    batch_size = 4
    dummy_latent_z = torch.randn(batch_size, latent_size)
    reconstructed_image = decoder(dummy_latent_z)

    print(f"Decoder instantiated successfully.")
    print(f"Input latent z shape: {dummy_latent_z.shape}")
    print(f"Reconstructed image shape: {reconstructed_image.shape}") # Expected: (batch_size, 1, 64, 64)

    # --- Print model summary (if torchsummary is available) ---
    # The input to the decoder is the latent vector, so input_size for summary is (latent_size,)
    try:
        print("\nDecoder Summary:")
        summary(decoder, input_size=(latent_size,))
    except ImportError:
        print("\n torchsummary not installed. Skipping summary.")
    except Exception as e:
        print(f"\nError during summary: {e}")

    # --- Verification of specific layer outputs (optional detailed check) ---
    # print("\n--- Detailed Layer Check ---")
    # test_mlp_out = decoder.mlp(dummy_latent_z)
    # print(f"MLP output shape: {test_mlp_out.shape}") # Expected: (batch_size, 128)
    # C_enc, H_enc, W_enc = decoder.encoder_conv_output_shape
    # test_reshaped = test_mlp_out.view(batch_size, C_enc, H_enc, W_enc)
    # print(f"Reshaped MLP output shape: {test_reshaped.shape}") # Expected: (batch_size, 32, 2, 2)

    # print("\nDeconv layers:")
    # current_tensor = test_reshaped
    # for i, layer in enumerate(decoder.deconv):
    #     current_tensor = layer(current_tensor)
    #     print(f"After deconv layer {i} ({type(layer).__name__}): {current_tensor.shape}")