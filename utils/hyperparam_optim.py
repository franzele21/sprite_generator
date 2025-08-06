import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from model.vae import *

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import tune
# from ray.tune.schedulers import ASHAScheduler

import os
import tempfile

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune import Checkpoint

from torchsummary import summary


# il faut mettre le chemin entier
assert os.path.exists(SPRITE_PATH:="/home/franzele/Desktop/sprite_generator/sprite_generator/sprites.csv"), "Le fichier de sprite n'est pas au bon endroit"

BATCH_SIZE = 64
EPOCH_SIZE = 512
TEST_SIZE = 256

def train_func(model:VAE, optimizer, train_loader, kld_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.train_model(train_loader, optimizer, int(EPOCH_SIZE/BATCH_SIZE)+1, device, kld_weight)
    return


def test_func(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    diff = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > TEST_SIZE:
                break
            data = data.to(device)
            recon_images, mu, log_var = model(data)

            loss, recon_loss, kl_loss = model.vae_loss(
                recon_images, data, mu, log_var, 1
            )

            print(f"\n\nLOSSES:\n{recon_loss=}\n{kl_loss=}\n\n")

            diff += loss

    return diff

def train_vae(cfg):
    train_loader = DataLoader(
        dataset = CSVDataset(SPRITE_PATH, 1000),
        batch_size=BATCH_SIZE,
        shuffle=True)
    test_loader = DataLoader(
        dataset = CSVDataset(SPRITE_PATH, 500),
        batch_size=BATCH_SIZE,
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_krnl_sz = int(cfg["f_krnl_sz"]/cfg["f_strd"])+1
    s_krnl_sz = int(cfg["s_krnl_sz"]/cfg["s_strd"])+1
    t_krnl_sz = int(cfg["t_krnl_sz"]/cfg["t_strd"])+1

    enc_conv_layers = (
        # (in_ch,           out_ch,             kernel,             stride,      padding)
        (1,                 cfg["f_chan_nb"],   cfg["f_krnl_sz"],   f_krnl_sz,  cfg["f_strd"]),
        (cfg["f_chan_nb"],  cfg["s_chan_nb"],   cfg["s_krnl_sz"],   s_krnl_sz,  cfg["s_strd"]),
        (cfg["s_chan_nb"],  cfg["t_chan_nb"],   cfg["t_krnl_sz"],   t_krnl_sz,  cfg["t_strd"])
    )
    
    enc_mlp_layers = (
        cfg["f_mlp_lyr_sz"], 
        cfg["s_mlp_lyr_sz"], 
        cfg["t_mlp_lyr_sz"],
        cfg["4_mlp_lyr_sz"]
    )

    original_img_dims = (64, 64)

    model = VAE(
        conv_layers_encoder_config=enc_conv_layers,
        mlp_layers_encoder_config=enc_mlp_layers,
        reparam_size=cfg["reparam_size"],
        num_resnet_blocks=cfg["num_resnet_blocks"],
        expansion_factor=cfg["expansion_factor"],
        rand_intensity=cfg["rand_intensity"],
        original_image_dims=original_img_dims
    )
    model.to(device)

    summary(model, (1, 64, 64))


    optimizer = optim.Adam(
        model.parameters(), lr=cfg["lr"])
    for i in range(10):
        train_func(model, optimizer, train_loader, cfg["kld_weight"])
        acc:torch.Tensor = test_func(model, test_loader)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if (i + 1) % 5 == 0:
                # This saves the model to the trial directory
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model.pth")
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Send the current training result back to Tune
            tune.report({"mean_loss": acc.item()}, checkpoint=checkpoint)

search_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    # f = first, s = sd, t = thrid
    # channel number
    "f_chan_nb": tune.randint(16, 65),
    "s_chan_nb": tune.randint(16, 65),
    "t_chan_nb": tune.randint(16, 65),
    # kernel size
    "f_krnl_sz": tune.randint(4, 33),
    "s_krnl_sz": tune.randint(4, 17),
    "t_krnl_sz": tune.randint(4, 9),
    # stride
    "f_strd": tune.randint(1, 11),
    "s_strd": tune.randint(1, 9),
    "t_strd": tune.randint(1, 7),
    # padding
    "f_pddng": tune.randint(0, 6),
    "s_pddng": tune.randint(0, 6),
    "t_pddng": tune.randint(0, 6),
    # MLP layer size
    "f_mlp_lyr_sz": tune.randint(16, 513),
    "s_mlp_lyr_sz": tune.randint(16, 513),
    "t_mlp_lyr_sz": tune.randint(16, 513),
    "4_mlp_lyr_sz": tune.randint(16, 513),

    "num_resnet_blocks": tune.randint(1, 10),
    "reparam_size": tune.randint(16, 513),

    "expansion_factor": tune.randint(2, 10),
    "rand_intensity": tune.uniform(0, 1),

    "kld_weight": tune.uniform(0, 1)
}

search_alg = OptunaSearch(metric="mean_loss", mode="min")
scheduler = ASHAScheduler(metric="mean_loss", mode="min", max_t=10, grace_period=1, reduction_factor=2)
trainable_with_resources = tune.with_resources(train_vae, {"cpu": 2})

tuner = tune.Tuner(
    trainable_with_resources,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=10,
        max_concurrent_trials=2
    ),
)
results = tuner.fit()


