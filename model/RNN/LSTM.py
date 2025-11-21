import torch
from torch import nn, optim
from tqdm import tqdm
import os
import random
import string
from datetime import datetime
import matplotlib.pyplot as plt


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
    
    def create_plot(self, data, ae, num_frames=5, decode_input_shape=(32,3,3)):
        """
        create a plot comparing input and reconstructed output from the LSTM
        data : Tensor [1, seq_len, latent_dim]
        ae : autoencoder model to decode the latent vectors
        num_frames : number of frames to display
        decode_input_shape : shape to reshape the latent vector before decoding
        """
        self.eval()
        ae.eval()

        with torch.no_grad():
            data = data.to(self.device)
            pred, _ = self(data)

            data = data.cpu()
            pred = pred.cpu()

            seq_len = data.size(1)
            step = max(1, seq_len // num_frames)
            selected_indices = list(range(0, seq_len, step))[:num_frames]

            fig, axes = plt.subplots(2, len(selected_indices), figsize=(len(selected_indices)*2, 4))

            for i, idx in enumerate(selected_indices):
                # original
                latent_vec = data[0, idx, :].unsqueeze(0)
                z_reshaped = latent_vec.reshape(1,32,3,3)
                recon_orig = ae.decode(z_reshaped).cpu().squeeze().numpy()
                axes[0, i].imshow(recon_orig, cmap='gray')
                axes[0, i].set_title(f"Original Frame {idx}")
                axes[0, i].axis('off')

                # reconstructed
                latent_vec_pred = pred[0, idx, :].unsqueeze(0)
                z_reshaped_pred = latent_vec_pred.reshape(1,32,3,3)
                recon_pred = ae.decode(z_reshaped_pred).cpu().squeeze().numpy()
                axes[1, i].imshow(recon_pred, cmap='gray')
                axes[1, i].set_title(f"Reconstructed Frame {idx}")
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.show()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_dir, "reconstructed")
            
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"reconstructed_{self.model_name}.png")

            plt.savefig(path, dpi=300, bbox_inches='tight')    
    

if __name__ == "__main__":
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader
    from Dataset import HomogeneousBatchDataset
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))      # .../models/autoencoder
    parent_dir = os.path.dirname(os.path.dirname(current_dir))    # .../project
    sys.path.append(parent_dir)
    from model.embedding.autoEncoder import autoAE, conv_layers_config

    df = pd.read_csv("sprites_latents.csv")
    # [pkmn_nb, idx]
    ref = df.iloc[:,:3]
    pkmn_nb = df.iloc[:, 0].values
    unique_pkmn = np.unique(pkmn_nb)
    np.random.shuffle(unique_pkmn)

    # latent space (864,)
    latent = df.iloc[:,3:]
    X = latent.values.astype('float32')

    # dictionnaire : pkmn_id -> liste de latents
    sequences_dict = defaultdict(list)

    for (pkmn_id, idx, mode), latent_vec in zip(ref.values, X):
        sequences_dict[pkmn_id,mode].append(torch.tensor(latent_vec))

    # split
    n_train = int(0.95 * len(unique_pkmn))
    train_pkmn = set(unique_pkmn[:n_train])
    test_pkmn = set(unique_pkmn[n_train:])
    # transformer en liste de séquences (chaque élément = séquence complète d'un Pokémon)
    train_sequences = [torch.stack(v) for k,v in sequences_dict.items() if k[0] in train_pkmn]
    train_sequences = [torch.cat([seq, seq], dim=0) for seq in train_sequences]
    train_sequences.sort(key=lambda x: x.shape[0])
    
    test_sequences = [torch.stack(v) for k,v in sequences_dict.items() if k[0] in test_pkmn]
    test_sequences = [torch.cat([seq, seq], dim=0) for seq in test_sequences]
    test_sequences.sort(key=lambda x: x.shape[0])
    # all_sequences = [torch.stack(v) for v in sequences_dict.values()]
    # all_sequences = [torch.cat([seq, seq], dim=0) for seq in all_sequences]
    # all_sequences.sort(key=lambda x: x.shape[0])

    train_by_len = defaultdict(list)
    for seq in train_sequences:
        train_by_len[seq.shape[0]].append(seq)
    test_by_len = defaultdict(list)
    for seq in test_sequences:
        test_by_len[seq.shape[0]].append(seq)

    batch_size = 3
    batches = []

    for length, sequences in train_by_len.items():
        for i in range(0, len(sequences), batch_size):
            print(i)
            batch = sequences[i:i+batch_size]

            # gérer batch incomplet
            if len(batch) < batch_size:
                while len(batch) < batch_size:
                    batch.append(batch[0].clone())
            
            batch_tensor = torch.stack(batch)  # shape: (batch_size, seq_len, 864)
            batches.append(batch_tensor)

    dataset = HomogeneousBatchDataset(train_by_len, batch_size=8)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    lstm = LSTM(input_size=288, hidden_size=156, num_layers=1)
    lstm.train_lstm(dataloader, n_epochs=50, lr=1e-3)
    ae = autoAE(conv_config=conv_layers_config, input_dim=64, load_path="./model/embedding/trained/autoAE_20251029_033430_qpsl.pt")
    lstm.create_plot(batches[0], ae, num_frames=5, decode_input_shape=(32,3,3))
    lstm.save()