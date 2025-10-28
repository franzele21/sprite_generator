import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader # For Dataset and DataLoader
import pandas as pd # Example for CSV handling in Dataset, not directly used by train_model
import torchvision
from torchvision import transforms


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

batch_size = 4
trainset = torchvision.datasets.MNIST(root='~/data/', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='~ /data/', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

conv_layers_config = [
    [
        [1, 16, 2, 1, 1], 
        [16, 24, 2, 1, 0], 
        [24, 32, 3, 1, 2]
    ], [
        [32, 48, 2, 1, 1], 
        [48, 64, 2, 1, 1]
    ]
]



class UNet(nn.Module):
    def __init__(self, conv_config, input_dims):
        super(UNet, self).__init__()

        self.encoder = []
        for conv_cfg in conv_config:
            print(conv_cfg)
            self.encoder.append(
                nn.Sequential(
                    *[[nn.Conv2d(*conv_cfg[i//3]),
                      nn.ReLU(),
                      nn.BatchNorm2d(conv_cfg[i//3][1])][i%3]
                      for i in range(len(conv_cfg)*3)]
                )
            )
        # on inverse la convolution
        decode_config = [x[::-1] for x in conv_config[::-1]]
        # on inverse les channels
        decode_config = [[[y[1], y[0], *y[2:]] for y in x] for x in decode_config]
        # on double le channel d'entré pour pouvoir concaténer
        decode_config = [[[y[0]*2, *y[1:]] if i==0 else y for i,y in enumerate(x)] for x in decode_config]

        self.decoder = []
        for decode_cfg in decode_config:
            self.decoder.append(
                nn.Sequential(
                    *[[nn.ConvTranspose2d(*decode_cfg[i//2]),
                       nn.ReLU()][i%2]
                       for i in range(len(decode_cfg)*2)]
                )
            )
    def forward(self, x):
        skip_connection = []
        for lyr in self.encoder:
            x = lyr(x)
            skip_connection.append(x)
        skip_connection = skip_connection[::-1]
        for i, lyr in enumerate(self.decoder):
            new_x = torch.concat([skip_connection[i], x], dim=1)
            x = lyr(new_x)
        y = F.sigmoid(x)
        return y
    
    def reproduction_train(self, trainset):
        pass

any(map(lambda x:x%1, conv_dim))