import torch
from torch import nn, optim
from tqdm import tqdm
import os
import random
import string
from datetime import datetime

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, load_path=None):
        super(LSTM,self).__init__()
        # name of the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        self.model_name = f"LSTM_{timestamp}_{rand_suffix}" 

        # model
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # if the ouput size is different than the input size
        self.predictor = nn.Linear(hidden_size,input_size)

        if load_path is not None:
            self.load(load_path)

        # select GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Using device: {self.device}")

    def forward(self, x, hidden=None):
        # x : DataLoader (batch, seq_len, lattent_space)
        output, (h_n, c_n) = self.lstm(x, hidden)

        predited = self.predictor(output)
        return predited, (h_n, c_n)
    
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
    
    def train_lstm(self, dataloader, n_epochs=50, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        progress =tqdm(range(n_epochs*len(dataloader)))
        self.train()
        result = []
        for epoch in range(n_epochs):
            total_loss=0

            for batch in dataloader:
                x = batch[0].to(self.device)

                '''
                WAR: ici la taille de sequence des elements doit etre identique dans le batch
                donc il faut trouver le nombre de frames qu'as chaque gif
                '''
                x_in = x[:,:-1,:]
                x_target = x[:,1:,:]

                pred, _ =self(x_in)
                #breakpoint()
                loss = criterion(pred, x_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                progress.update()

            tqdm.write(f"Epoch [{epoch+1}/{n_epochs}] - loss: {total_loss/len(dataloader):.6f}")
            result.append([f"Epoch [{epoch+1}/{n_epochs}]", total_loss/len(dataloader)])
        
        return result
    
    def save(self, folder_name="trained"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, folder_name)
        
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.model_name}.pt")
        
        torch.save(self.state_dict(), path)
        print(f"Model saved at: {path}")
        return path
    
if __name__ == "__main__":
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader
    from Dataset import HomogeneousBatchDataset

    df = pd.read_csv("clean_sprites_latents.csv")
    # [pkmn_nb, idx]
    ref = df.iloc[:,:2]
    # latent space (864,)
    latent = df.iloc[:,2:]
    X = latent.values.astype('float32')

    # dictionnaire : pkmn_id -> liste de latents
    sequences_dict = defaultdict(list)

    for (pkmn_id, idx), latent_vec in zip(ref.values, X):
        sequences_dict[pkmn_id].append(torch.tensor(latent_vec))

    # transformer en liste de séquences (chaque élément = séquence complète d'un Pokémon)
    all_sequences = [torch.stack(v) for v in sequences_dict.values()]
    all_sequences.sort(key=lambda x: x.shape[0])

    seq_by_len = defaultdict(list)
    for seq in all_sequences:
        seq_by_len[seq.shape[0]].append(seq)

    batch_size = 8
    batches = []

    for length, sequences in seq_by_len.items():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]

            # gérer batch incomplet
            if len(batch) < batch_size:
                while len(batch) < batch_size:
                    batch.append(batch[0].clone())
            
            batch_tensor = torch.stack(batch)  # shape: (batch_size, seq_len, 864)
            batches.append(batch_tensor)

    dataset = HomogeneousBatchDataset(seq_by_len, batch_size=8)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    lstm = LSTM(input_size=864, hidden_size=512, num_layers=3)
    lstm.train_lstm(dataloader, n_epochs=50, lr=1e-3)
    lstm.save()