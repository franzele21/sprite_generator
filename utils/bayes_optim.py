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

from model.vae import *

from ax.service.managed_loop import optimize

# Suppress warnings for cleaner output
# warnings.filterwarnings("ignore", category=UserWarning)


# # Fixed model parameters that won't be tuned
# FIXED_PARAMETERS = {
#     "conv_layers_encoder_config": (
#         (1, 16, 16, 8, 0),
#         (16, 24, 8, 4, 2),
#         (24, 32, 2, 1, 1)
#     ),
#     "encoder_conv_output_shape": (32, 2, 2),
#     "original_image_dims": (64, 64)
# }
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# class CSVDataset(Dataset):
#     def __init__(self,
#                  csv_path: str,
#                  nrows: int,
#                  transform=None):
#         self.transform = transform
#         self.nrows = nrows
#         self.csv_path = csv_path
#         self._load_data()

#     def _load_data(self):
#         print(f"Loading up to {self.nrows} samples from {self.csv_path}...")
        
#         # Determine total rows for sampling
#         total_rows = sum(1 for _ in open(self.csv_path, 'r')) - 1
#         nrows_to_read = min(self.nrows, total_rows) if self.nrows > 0 else total_rows
        
#         skip_indices = []
#         if nrows_to_read < total_rows:
#             skip_indices = sorted(np.random.choice(
#                 range(1, total_rows + 1),
#                 size=total_rows - nrows_to_read,
#                 replace=False
#             ))

#         pixel_data = pd.read_csv(
#             self.csv_path,
#             usecols=list(range(3, 3 + 64*64)),
#             skiprows=skip_indices,
#             header=0
#         )
        
#         self.pixel_data = torch.tensor(pixel_data.values, dtype=torch.float32)
#         self.num_samples = len(self.pixel_data)
#         print(f"Loaded {self.num_samples} samples with shape {self.pixel_data.shape}")

#     def __len__(self) -> int:
#         return self.num_samples

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         pixel_data = self.pixel_data[idx]
#         image_tensor = pixel_data.view(1, 64, 64)
        
#         # Normalize to [0, 1]
#         max_val = image_tensor.max()
#         if max_val > 0:
#             image_tensor = image_tensor / max_val
        
#         if self.transform:
#             image_tensor = self.transform(image_tensor)
        
#         return image_tensor


# def init_vae(parameterization: dict):
#     """Initializes the VAE model from a dictionary of hyperparameters."""
    
#     # Construct the MLP layers configuration from hyperparameters
#     mlp_layers = tuple([parameterization.get(f"mlp_layer_{i}", 128) for i in range(parameterization.get("num_mlp_layers", 2))])
    
#     model = VAE(
#         conv_layers_encoder_config=FIXED_PARAMETERS["conv_layers_encoder_config"],
#         mlp_layers_encoder_config=mlp_layers,
#         reparam_size=parameterization.get("latent_size", 64),
#         encoder_conv_output_shape=FIXED_PARAMETERS["encoder_conv_output_shape"],
#         original_image_dims=FIXED_PARAMETERS["original_image_dims"]
#     )
#     return model

# def evaluate_model_loss(net: VAE, data_loader: DataLoader, device: torch.device, kld_weight: float):
#     """Evaluates the VAE, returning the average total loss."""
#     net.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for images in data_loader:
#             images = images.to(device)
#             recon_images, mu, log_var = net(images)
#             loss, _, _ = net.vae_loss(recon_images, images, mu, log_var, kld_weight)
#             total_loss += loss.item()
            
#     return total_loss / len(data_loader)

# def train_evaluate_vae(parameterization: dict, trial_index: int):
#     """
#     The main function for Ax. It creates, trains, and evaluates a VAE
#     for one set of hyperparameters.
#     """
#     # 1. Initialize the VAE with the given parameters
#     net = init_vae(parameterization)
#     net.to(device)
    
#     # 2. Get hyperparameters for training
#     lr = parameterization.get("lr", 1e-3)
#     kld_weight = parameterization.get("kld_weight", 1.0)
#     # Use a fixed number of epochs for each trial to ensure a fair comparison
#     epochs = 15 
    
#     # 3. Create the optimizer
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
#     # 4. Train the model
#     net.train_model(
#         data_loader=train_loader,
#         optimizer=optimizer,
#         epochs=epochs,
#         device=device,
#         kld_weight=kld_weight,
#     )
    
#     # 5. Evaluate the model on the validation set
#     loss = evaluate_model_loss(
#         net=net,
#         data_loader=val_loader,
#         device=device,
#         kld_weight=kld_weight
#     )
    
#     # Ax requires a dictionary with the objective name and a (mean, SEM) tuple.
#     # We report 0.0 for the standard error of the mean (SEM) as we are not computing it.
#     return {"loss": (loss, 0.0)}

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # 3. MAIN EXECUTION BLOCK
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    try:
        # Load the full dataset once
        # Using a smaller sample (5000) for faster optimization runs
        full_dataset = CSVDataset("./sprites.csv", nrows=5000)
        
        # Split into training and validation sets
        train_size = int(0.85 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create DataLoaders (assigning to global variables)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
    except FileNotFoundError:
        print("Error: 'sprites.csv' not found. Please place the data file in the correct directory.")
        exit()
    
    # Define the hyperparameter search space for Ax
    search_space = [
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "latent_size", "type": "range", "bounds": [16, 128]},
        {"name": "kld_weight", "type": "range", "bounds": [0.1, 5.0]},
        {"name": "num_mlp_layers", "type": "range", "bounds": [1, 4]},
        # Dynamic MLP layer sizes based on num_mlp_layers
        {"name": "mlp_layer_0", "type": "range", "bounds": [64, 256]},
        {"name": "mlp_layer_1", "type": "range", "bounds": [64, 256]},
        {"name": "mlp_layer_2", "type": "range", "bounds": [32, 128]},
        {"name": "mlp_layer_3", "type": "range", "bounds": [32, 128]},
    ]
    
    print("--- Starting Bayesian Optimization for VAE ---")
    
    # Create a wrapper for the evaluation function to include the trial_index
    # Ax's ManagedLoop passes the parameterization dict, but we also want to track the trial number for logging
    class EvaluationWrapper:
        def __init__(self):
            self.trial_index = 0
        def __call__(self, params):
            result = train_evaluate_vae(params, self.trial_index)
            self.trial_index += 1
            return result

    evaluation_function_wrapper = EvaluationWrapper()
    
    best_parameters, values, experiment, model = optimize(
        parameters=search_space,
        evaluation_function=evaluation_function_wrapper,
        objective_name='loss',
        minimize=True,
        total_trials=25 # Number of different hyperparameter sets to try
    )
    
    print("\n--- Optimization Finished ---")
    print("Best Parameters Found:")
    print(best_parameters)
    
    means, covariances = values
    print("\nBest Objective Mean (Loss) and Covariance:")
    print(f"Loss: {means['loss']:.4f}")
    # print(covariances) # Can be verbose, uncomment if needed