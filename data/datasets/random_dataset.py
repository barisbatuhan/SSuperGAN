import torch
import numpy as np
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    
    def __init__(self, in_dim, target_dim):
        self.in_dim = in_dim
        self.target_dim = target_dim
        
    def __len__(self):
        return 100 # * np.prod(self.in_dim)
    
    def __getitem__(self, idx): 
        x = torch.randn(*self.in_dim)
        y = torch.randn(*self.target_dim)
        return x, y