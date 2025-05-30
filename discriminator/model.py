"""
Possible Discriminator of this project.
Could be used to enhance the performance of the generator.
Is a classifier, that will say if the input is a Pokémon or not.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import numpy as np
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 nrows: int,
                 transform=None,
                 chunk_size=10000):

        self.transform = transform
        self.pixel_data = []
        self.nrows = nrows
        self.chunk_size = chunk_size
        self.csv_path = csv_path

        # Calculate the total number of rows in the CSV file
        self.total_rows = sum(1 for _ in open(csv_path, 'r')) - 1  # Subtract 1 for the header

        # Ensure nrows does not exceed the total number of rows
        self.nrows = min(self.nrows, self.total_rows)

        # Sample row indices
        self.sampled_indices = np.random.choice(self.total_rows, self.nrows, replace=False)

        # Load the sampled rows
        self._load_sampled_rows()

    def _load_sampled_rows(self):
        # Read the CSV file in chunks and collect the sampled rows
        for chunk in tqdm(pd.read_csv(
            self.csv_path,
            usecols=list(range(3, 3+64*64)),
            header=0,
            chunksize=self.chunk_size
        )):
            # Find the intersection of sampled indices and the current chunk
            chunk_start = len(self.pixel_data)
            chunk_end = chunk_start + len(chunk)
            chunk_indices = [i - chunk_start for i in self.sampled_indices if chunk_start <= i < chunk_end]

            if chunk_indices:
                sampled_chunk = chunk.iloc[chunk_indices]
                self.pixel_data.append(sampled_chunk)

        # Concatenate all sampled chunks into a single DataFrame
        self.pixel_data = pd.concat(self.pixel_data, ignore_index=True)

        # Convert to tensor
        self.pixel_data = torch.tensor(self.pixel_data.values, dtype=torch.float32)
        self.num_samples = len(self.pixel_data)  # Number of rows loaded

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a sample (image tensor) from the dataset at the given index.
        """
        flattened_image_data = self.pixel_data[idx]

        image_tensor = flattened_image_data.flatten()
        image_tensor /= 255

        image_tensor = image_tensor.view(1, 64, 64)

        y = 1
        if np.random.random() < 0.5:
            image_tensor += 0.5*torch.randn(1, 64, 64)
            y = 0

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, y



class Discriminator(nn.Module):
    def __init__(self, 
                 conv_layers: tuple[tuple[int]], 
                 mlp_layers: tuple[int], 
                 ):
        super().__init__()

        self.conv = nn.Sequential(
            *[[nn.Conv2d(*conv_layers[i//3]), 
               nn.BatchNorm2d(conv_layers[i//3][1]),
               nn.ReLU(),
               ][i%3] for i in range(len(conv_layers)*3)]
        )
        self.mlp = nn.Sequential(
            *[[nn.Linear(mlp_layers[i//2], mlp_layers[i//2+1]), 
               nn.ReLU()][i%2] for i in range(len(mlp_layers)*2-2)]
        )
        self.last_layer = nn.Sequential(
            nn.Linear(mlp_layers[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.conv(x)
        z = torch.flatten(z, start_dim=1)
        z = self.mlp(z)
        y = self.last_layer(z)

        return y

    def train(self, 
              data_loader,
              optimizer: optim.Optimizer, 
              epoch=1000):

        losses = []
        for i in tqdm(range(epoch)):
            losses.append(0)
            for X, y in data_loader:
                optimizer.zero_grad()

                pred = self(X)

                y = y.unsqueeze(1).float()
                # print(y, pred)

                loss = F.binary_cross_entropy(pred, y)

                loss.backward()
                optimizer.step()

                losses[-1] += loss.item()
            print(losses[-1])
        return losses

    def save(self, path="discriminator.pth"):
        torch.save(self.state_dict(), path)
    
    def load(self, path="discriminator.pth"):
        self.load_state_dict(torch.load(path))

    
if __name__ == "__main__":
    from torchsummary import summary
    import matplotlib.pyplot as plt

    conv_layers = (
        (1, 16, 16, 8, 0),
        (16, 24, 8, 4, 2),
        (24, 32, 2, 1, 1)
    )
    mlp_layers = (128, 64, 32)

    discri = Discriminator(conv_layers, mlp_layers)
    print(discri)
    summary(discri, (1, 64, 64), device="cpu")

    dummy_dataset = CSVDataset(
        "./sprites.csv",
        10_000
    )
    print(len(dummy_dataset))

    tmp_iter = iter(dummy_dataset)
    f, ax = plt.subplots(2, 2)
    types = []
    for i in range(2):
        for j in range(2):
            tmp = next(tmp_iter)
            ax[i][j].imshow(tmp[0].view(64, 64))
            types.append(tmp[1])
    print(types)
    del tmp_iter

    train_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True, num_workers=0) 

    optimizer = torch.optim.Adam(discri.parameters(), 1e-5)

    discri.train(
        data_loader=train_loader,
        optimizer=optimizer,
        epoch=50
    )

    discri.save()