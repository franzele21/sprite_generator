from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from math import floor
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import string
from datetime import datetime
import numpy as np

conv_layers_config = [
        [1, 8, 2, 1, 0], 
        [8, 16, 3, 2, 1], 
        [16, 24, 4, 2, 2],
        [24, 32, 4, 1, 2],
        [32, 32, 4, 2, 2],
        [32, 32, 4, 4, 1]
]



class autoAE(nn.Module):
    def __init__(self, conv_config, input_dim, load_path=None):
        super(autoAE, self).__init__()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        self.model_name = f"autoAE_{timestamp}_{rand_suffix}" 

        conv_calc = lambda dim_in, kernel_size, stride, padding: 1 + (floor(dim_in) + 2*padding - (kernel_size-1)-1)/stride
        self.encoder = nn.Sequential(
            *[[nn.Conv2d(*conv_config[i//3]),
               nn.BatchNorm2d(conv_config[i//3][1]),
               nn.LeakyReLU(0.1)][i%3] for i in range(len(conv_config)*3)])
            
        conv_dim = [input_dim]
        for i,lyr in enumerate(conv_config):
            conv_dim.append(conv_calc(conv_dim[-1], *lyr[2:]))
        need_output_padding = list(map(lambda x:not x%1==0, conv_dim))[::-1][:-1]
        if any(need_output_padding):
            print(f"Warning : this architecture needs output_paddings on the expanding side, on layers : {', '.join([str(i) for i,x in enumerate(need_output_padding) if x])}")

        deconv_config = conv_config[::-1]
        deconv_config = [[x[1],x[0],*x[2:]] for x in deconv_config]
        for i,x in enumerate(need_output_padding):
            if x:
                deconv_config[i].append(1)
        self.decoder = nn.Sequential(
            *[[nn.ConvTranspose2d(*deconv_config[i//3]),   
               nn.BatchNorm2d(deconv_config[i//3][1]),
               nn.LeakyReLU(0.1)][i%3] for i in range(len(deconv_config)*3)])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Using device: {self.device}")

        if load_path is not None:
            self.load(load_path)
            
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        y = F.sigmoid(y)
        return y
    
    def fit(self, dataLoader:DataLoader, lossFunc:str="mseloss",
            opt:str ="adam", nepochs:int=20):

        crit_methods={
            "mseloss":nn.MSELoss,
            "l1loss":nn.L1Loss,
            "cel":nn.CrossEntropyLoss
        }
        if lossFunc not in crit_methods:
            lossFunc = "mseloss"
        criterion = crit_methods[lossFunc]()

        optim_methods = {
            "adam":optim.Adam,
            "sgd":optim.SGD
        }
        if opt not in optim_methods:
            opt = "adam"
        optimizer = optim_methods[opt](self.parameters(), lr=1e-3)
        
        progress =tqdm(range(nepochs*len(dataLoader)))
        result=[]
        for epoch in range(nepochs):
            self.train()
            total_loss = 0

            for batch, _ in dataLoader:
                batch = batch.to(self.device)

                outputs = self(batch)
                loss = criterion(outputs, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress.update()
                progress.refresh()

            tqdm.write(f"Epoch [{epoch+1}/{nepochs}], Loss: {total_loss/len(dataLoader):.6f}")
            result.append([f"Epoch [{epoch+1}/{nepochs}]", total_loss/len(dataLoader)])
        progress.close()
        return result
    
    def encode(self, x):
        """
        return latent vector
        x : Tensor [B, C, H, W]
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            z = self.encoder(x)
        return z

    def decode(self, z):
        """
        decode from latent space
        z : Tensor [B, latent_dim, H', W']
        """
        self.eval()
        with torch.no_grad():
            z = z.to(self.device)
            y = self.decoder(z)
            y = torch.sigmoid(y)
        return y

    def get_latent_space(self, data):
        """
        take a dataloader and return the latent space representation
        data : DataLoader
        """
        self.eval()
        latent_vectors = []
        with torch.no_grad():
            for batch, _ in data:
                batch = batch.to(self.device)
                z = self.encoder(batch)
                latent_vector = nn.Flatten()(z)
                latent_vectors.append(latent_vector.cpu().numpy())
        
        latent_vectors = np.vstack(latent_vectors)
        return latent_vectors
    
    def save(self, folder_name="trained"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, folder_name)
        
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.model_name}.pt")
        
        torch.save(self.state_dict(), path)
        print(f"Model saved at: {path}")
        return path
    
    def load(self, path):
        """
            load a trained model from .pt file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier '{path}' est introuvable.")
        
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()
        print(f"Model loaded from : {path}")
    
    def predict(self, data, batch_size=32, folder_name="reconstructed"):
        self.eval()
        with torch.no_grad():
            if isinstance(data, DataLoader):
                imgs, _ = next(iter(data))
                imgs = imgs.to(self.device)
                outputs = self(imgs)
            else:  # ca doit etre un tensor au minimum
                if data.ndim == 3:  
                    data = data.unsqueeze(0)

                if data.size(0) > 5:
                    idx = torch.randperm(data.size(0))[:5]
                    data = data[idx]
                
                for i in range(0, data.size(0), batch_size):
                    imgs = data[i:i+batch_size].to(self.device)
                    outputs = self(imgs)

        imgs = imgs.cpu().numpy()
        outputs = outputs.cpu().numpy()

        for i in range(5):
            plt.subplot(2,5,i+1)
            plt.imshow(imgs[i,0], cmap='gray')
            plt.axis('off')
            plt.ylabel("Original")
            plt.subplot(2,5,i+6)
            plt.imshow(outputs[i,0], cmap='gray')
            plt.axis('off')
            plt.ylabel("Reconstruit")
        plt.show()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, folder_name)
        
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"reconstructed_{self.model_name}.png")

        plt.savefig(path, dpi=300, bbox_inches='tight')

        return 

if __name__ == "__main__":
    # read data
    df = pd.read_csv("./clean_sprites.csv")
    # orientation = df.iloc[:, 2].values
    df = df.iloc[:,2:]
    X = df.values.astype('float32')

    # process data
    X = X.reshape(-1, 1, 64, 64) / 255.0
    # for i, mode in enumerate(orientation):
    #     if mode == 1:
    #         X[i,0] = np.rot90(X[i,0], k=1)   # rotation 90° à gauche
    #     elif mode == 2:
    #         X[i,0] = np.rot90(X[i,0], k=3)
    x_tensor = torch.tensor(X, dtype=torch.float32)


    # create batch
    dataset = TensorDataset(x_tensor, x_tensor)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

    # create and train autoencoder
    # ae = autoAE(conv_config=conv_layers_config, input_dim=64)

    # ae.fit(dataLoader,lossFunc="l1loss", opt="adam")
    # ae.save()
    # ae.predict(x_tensor)

    # load and predict
    ae2 = autoAE(conv_config=conv_layers_config, input_dim=64, load_path="./model/embedding/trained/autoAE_20251026_194448_mm34.pt")
    ae2.fit(dataLoader,lossFunc="l1loss", opt="adam", nepochs=10)
    ae2.save()
    ae2.predict(x_tensor)
    # print(len(dataLoader))
    # print(ae2.get_latent_space(dataLoader).shape)
    # latent = ae2.predict(x_tensor)
    # dl = DataLoader(latent.reshape(1000,24,6,6), batch_size=32, shuffle=False)
    # for b,_ in dl:
    #     print(ae2.decode(b).shape)