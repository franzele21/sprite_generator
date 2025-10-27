from torch.utils.data import Dataset
import torch

class HomogeneousBatchDataset(Dataset):
    def __init__(self, seq_by_len, batch_size=8):
        self.batches = []
        for length, sequences in seq_by_len.items():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                # si batch incomplet, on peut juste l’accepter
                self.batches.append(torch.stack(batch))
                
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        return self.batches[idx]  # shape : (batch_size, seq_len, 864)
