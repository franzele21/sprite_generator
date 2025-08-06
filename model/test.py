import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import time

from ax.service.managed_loop import optimize

import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Fixed model parameters that won't be tuned
FIXED_PARAMETERS = {
    "conv_layers_encoder_config": (
        (1, 16, 3, 2, 1),    # Changed from 16x16 kernel to 3x3
        (16, 24, 3, 2, 1),   # More reasonable kernel sizes
        (24, 32, 3, 2, 1)    # Standard conv architecture
    ),
    "encoder_conv_output_shape": (32, 8, 8),  # Updated for new conv layers
    "original_image_dims": (64, 64)
}

# =============================================================================
# ENCODER
# =============================================================================

class Encoder(nn.Module):
    """
    Improved Encoder with proper architecture and reparameterization
    """
    def __init__(self, 
                 conv_layers: tuple[tuple[int]], 
                 mlp_layers: tuple[int], 
                 reparam_size: int,
                 randomize: bool = True,
                 return_mean_logvar: bool = False,
                 rand_intensity: float = 1.0  # Changed default to 1.0 for proper VAE
                 ):
        super().__init__()
        self.randomize = randomize
        self.return_mean_logvar = return_mean_logvar
        self.rand_intensity = rand_intensity
        self.reparam_size = reparam_size

        # Build convolutional layers
        conv_modules = []
        for layer_params in conv_layers:
            conv_modules.extend([
                nn.Conv2d(*layer_params),
                nn.BatchNorm2d(layer_params[1]),
                nn.ReLU(inplace=True)
            ])
        self.conv = nn.Sequential(*conv_modules)
        
        # Calculate conv output size for first MLP layer
        with torch.no_grad():
            dummy_input = torch.randn(1, conv_layers[0][0], 64, 64)
            conv_output = self.conv(dummy_input)
            self.conv_output_size = conv_output.numel()

        # Build MLP layers - first layer connects to flattened conv output
        mlp_modules = []
        prev_size = self.conv_output_size
        
        for hidden_size in mlp_layers:
            mlp_modules.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        self.mlp = nn.Sequential(*mlp_modules)

        # Reparameterization layers
        self.mean_layer = nn.Linear(prev_size, reparam_size)
        self.logvar_layer = nn.Linear(prev_size, reparam_size)
    
    def forward(self, x):
        # Convolutional feature extraction
        z = self.conv(x)
        z = torch.flatten(z, start_dim=1)
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
        Proper reparameterization trick implementation
        """
        if self.randomize:
            # Clamp log_var to prevent numerical instability
            log_var = torch.clamp(log_var, -10, 10)
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std) * self.rand_intensity
            z = mean + std * epsilon
        else:
            z = mean
        return z

# =============================================================================
# DECODER
# =============================================================================

class Decoder(nn.Module):
    """
    Improved Decoder with proper upsampling and architecture
    """
    def __init__(self,
                 conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...],
                 reparam_size: int,
                 encoder_conv_output_shape: tuple[int, int, int],
                 original_image_dims: tuple[int, int]):
        super().__init__()
        
        self.encoder_conv_output_shape = encoder_conv_output_shape
        self.reparam_size = reparam_size
        self.original_image_dims = original_image_dims
        
        # Calculate the size needed to reshape to conv tensor
        C, H, W = encoder_conv_output_shape
        self.conv_input_size = C * H * W
        
        # Build MLP layers in reverse order
        mlp_sizes = [reparam_size] + list(reversed(mlp_layers_encoder_config)) + [self.conv_input_size]
        
        mlp_modules = []
        for i in range(len(mlp_sizes) - 1):
            mlp_modules.extend([
                nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]),
                nn.BatchNorm1d(mlp_sizes[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2) if i < len(mlp_sizes) - 2 else nn.Identity()  # No dropout on last layer
            ])
        
        self.mlp = nn.Sequential(*mlp_modules)
        
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
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ])
            else:
                # Final layer - output image
                in_ch, out_ch = layer_params[1], layer_params[0]
                kernel_size = layer_params[2]
                stride = layer_params[3]
                padding = layer_params[4]
                
                deconv_modules.extend([
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
                    nn.Sigmoid()  # Output activation
                ])
        
        self.deconv = nn.Sequential(*deconv_modules)
        
        # Final adjustment layer if needed
        self.final_adjust = nn.AdaptiveAvgPool2d(original_image_dims)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        
        # MLP part
        x = self.mlp(z)
        
        # Reshape to conv tensor
        C, H, W = self.encoder_conv_output_shape
        x = x.view(batch_size, C, H, W)
        
        # Deconvolutional part
        x = self.deconv(x)
        
        # Ensure correct output size
        if x.shape[-2:] != self.original_image_dims:
            x = self.final_adjust(x)
        
        return x

# =============================================================================
# DATASET
# =============================================================================

class CSVDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 nrows: int,
                 transform=None):
        self.transform = transform
        self.nrows = nrows
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        print(f"Loading up to {self.nrows} samples from {self.csv_path}...")
        
        # More efficient loading
        try:
            # Read a small chunk first to get total rows
            chunk = pd.read_csv(self.csv_path, nrows=1)
            total_cols = len(chunk.columns)
            
            # Assume pixel data starts from column 3 and is 64x64 = 4096 pixels
            pixel_cols = list(range(3, min(3 + 64*64, total_cols)))
            
            if self.nrows > 0:
                # Sample random rows
                skiprows = lambda i: i > 0 and np.random.random() > (self.nrows / 10000)  # Approximate sampling
                df = pd.read_csv(self.csv_path, usecols=pixel_cols, skiprows=skiprows)
                df = df.head(self.nrows)  # Ensure we don't exceed nrows
            else:
                df = pd.read_csv(self.csv_path, usecols=pixel_cols)
            
            self.pixel_data = torch.tensor(df.values, dtype=torch.float32)
            
            # Normalize to [0, 1] globally
            if self.pixel_data.max() > 1:
                self.pixel_data = self.pixel_data / 255.0
            
            self.num_samples = len(self.pixel_data)
            print(f"Loaded {self.num_samples} samples with shape {self.pixel_data.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        pixel_data = self.pixel_data[idx]
        image_tensor = pixel_data.view(1, 64, 64)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor

# =============================================================================
# VAE MODEL
# =============================================================================

class VAE(nn.Module):
    def __init__(self, 
                 conv_layers_encoder_config: tuple[tuple[int, ...], ...],
                 mlp_layers_encoder_config: tuple[int, ...],
                 reparam_size: int,
                 encoder_conv_output_shape: tuple[int, int, int],
                 original_image_dims: tuple[int, int]):
        super().__init__()
        
        self.encoder = Encoder(
            conv_layers_encoder_config, 
            mlp_layers_encoder_config, 
            reparam_size,
            randomize=True,
            return_mean_logvar=True,
            rand_intensity=1.0
        )
        
        self.decoder = Decoder(
            conv_layers_encoder_config, 
            mlp_layers_encoder_config, 
            reparam_size,
            encoder_conv_output_shape,
            original_image_dims
        )
        
        # Store image dimensions
        self.img_c = conv_layers_encoder_config[0][0]
        self.img_h, self.img_w = original_image_dims
        
        print(f"VAE initialized for images: {self.img_c}x{self.img_h}x{self.img_w}")
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and decoder"""
        z, mean, log_var = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, log_var
    
    def vae_loss(self, recon_x, x, mu, log_var, kld_weight=1.0):
        """Calculate VAE loss with proper weighting"""
        batch_size = x.size(0)
        
        # Reconstruction loss - use MSE or BCE depending on data
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
        
        # KL divergence loss with numerical stability
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        
        # Total loss
        total_loss = recon_loss + kld_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_model(self, 
                    data_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    epochs: int, 
                    device: torch.device,
                    kld_weight: float = 1.0,
                    verbose: bool = True):
        """Train the VAE model with improved training loop"""
        self.train()
        self.to(device)
        
        if verbose:
            print(f"Starting training on {device} for {epochs} epochs...")
        
        for epoch in tqdm(range(epochs), leave=False):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            num_batches = 0
            
            # Use tqdm only if verbose
            loader = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) if verbose else data_loader
            
            for batch_data in loader:
                # Handle different data formats
                if isinstance(batch_data, (list, tuple)):
                    images = batch_data[0]
                else:
                    images = batch_data
                
                images = images.to(device)
                
                # Skip batch if wrong shape
                if images.dim() != 4 or images.shape[1:] != (self.img_c, self.img_h, self.img_w):
                    continue
                
                # Forward pass
                optimizer.zero_grad()
                recon_images, mu, log_var = self(images)
                
                # Calculate loss
                loss, recon_loss, kl_loss = self.vae_loss(
                    recon_images, images, mu, log_var, kld_weight
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Accumulate losses
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                num_batches += 1
                
                # Update progress bar
                if verbose and isinstance(loader, tqdm):
                    loader.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Recon': f'{recon_loss.item():.4f}',
                        'KL': f'{kl_loss.item():.4f}'
                    })
            
            # Epoch summary
            if num_batches > 0 and verbose:
                avg_loss = epoch_loss / num_batches
                avg_recon = epoch_recon_loss / num_batches
                avg_kl = epoch_kl_loss / num_batches
                
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")
        
        if verbose:
            print("Training completed!")

# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def init_vae(parameterization: dict):
    """Initializes the VAE model from a dictionary of hyperparameters."""
    
    # Construct the MLP layers configuration from hyperparameters
    num_layers = parameterization.get("num_mlp_layers", 2)
    mlp_layers = tuple([parameterization.get(f"mlp_layer_{i}", 128) for i in range(num_layers)])
    
    model = VAE(
        conv_layers_encoder_config=FIXED_PARAMETERS["conv_layers_encoder_config"],
        mlp_layers_encoder_config=mlp_layers,
        reparam_size=parameterization.get("latent_size", 64),
        encoder_conv_output_shape=FIXED_PARAMETERS["encoder_conv_output_shape"],
        original_image_dims=FIXED_PARAMETERS["original_image_dims"]
    )
    return model

def evaluate_model_loss(net: VAE, data_loader: DataLoader, device: torch.device, kld_weight: float):
    """Evaluates the VAE, returning the average total loss."""
    net.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                images = batch_data[0]
            else:
                images = batch_data
                
            images = images.to(device)
            
            # Skip invalid batches
            if images.dim() != 4:
                continue
                
            recon_images, mu, log_var = net(images)
            loss, _, _ = net.vae_loss(recon_images, images, mu, log_var, kld_weight)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train_evaluate_vae(parameterization: dict, trial_index: int = 0):
    """
    The main function for Ax optimization
    """
    try:
        # Initialize the VAE with the given parameters
        net = init_vae(parameterization)
        net.to(device)
        
        # Get hyperparameters for training
        lr = parameterization.get("lr", 1e-3)
        kld_weight = parameterization.get("kld_weight", 1.0)
        epochs = 10  # Reduced for faster optimization
        
        # Create the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        
        # Train the model
        net.train_model(
            data_loader=train_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            kld_weight=kld_weight,
            verbose=False  # Reduce verbosity during optimization
        )
        
        # Evaluate the model on the validation set
        loss = evaluate_model_loss(
            net=net,
            data_loader=val_loader,
            device=device,
            kld_weight=kld_weight
        )
        
        print(f"Trial {trial_index}: Loss = {loss:.4f}")
        
        return {"loss": (loss, 0.0)}
    
    except Exception as e:
        print(f"Trial {trial_index} failed: {e}")
        return {"loss": (float('inf'), 0.0)}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load the dataset
    full_dataset = CSVDataset("./sprites.csv", nrows=4000)  # Reduced for faster testing
    
    # Split into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        
    
    # Define the hyperparameter search space for Ax
    search_space = [
        {"name": "lr", "type": "range", "bounds": [1e-4, 5e-3], "log_scale": True, "value_type": "float"},
        {"name": "latent_size", "type": "range", "bounds": [16, 128], "value_type": "int"},
        {"name": "kld_weight", "type": "range", "bounds": [0.1, 2.0], "value_type": "float"},
        {"name": "num_mlp_layers", "type": "range", "bounds": [2, 4], "value_type": "int"},
       # {"name": "skip_mlp_nb", "type": "range", ""}
        {"name": "mlp_layer_0", "type": "range", "bounds": [64, 256], "value_type": "int"},
        {"name": "mlp_layer_1", "type": "range", "bounds": [32, 128], "value_type": "int"},
        {"name": "mlp_layer_2", "type": "range", "bounds": [32, 128], "value_type": "int"},
        {"name": "mlp_layer_3", "type": "range", "bounds": [16, 64], "value_type": "int"},
    ]
    
    print("--- Starting Bayesian Optimization for VAE ---")
    
    # Create evaluation wrapper
    class EvaluationWrapper:
        def __init__(self):
            self.trial_index = 0
        
        def __call__(self, params):
            result = train_evaluate_vae(params, self.trial_index)
            self.trial_index += 1
            return result

    evaluation_function_wrapper = EvaluationWrapper()
    
    # Run optimization
    best_parameters, values, experiment, model = optimize(
        parameters=search_space,
        evaluation_function=evaluation_function_wrapper,
        objective_name='loss',
        minimize=True,
        total_trials=30  # Reduced for faster testing
    )
    
    print("\n--- Optimization Finished ---")
    print("Best Parameters Found:")
    for key, value in best_parameters.items():
        print(f"  {key}: {value}")
    
    means, covariances = values
    print(f"\nBest Loss: {means['loss']:.4f}")
    
    # Train final model with best parameters
    print("\n--- Training Final Model ---")
    final_model = init_vae(best_parameters)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_parameters['lr'])
    
    final_model.train_model(
        data_loader=train_loader,
        optimizer=final_optimizer,
        epochs=250,  # More epochs for final model
        device=device,
        kld_weight=best_parameters['kld_weight'],
        verbose=True
    )
    
    # Save the final model
    final_model.eval()
    torch.save(final_model.state_dict(), f"best_vae_{time.strftime('%Y%m%d_%H%M%S')}.pth")
    print("Final model saved!")

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
    
    make_images(final_model, create_dataloader())