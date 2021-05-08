import os
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

from data.augment import read_image

class GoldenFacesDataset(Dataset):
    
    def __init__(self, folder_path :str, face_dim :int, shuffle :bool=True, augment :bool=True, limit_size :int=-1):
        self.dim = face_dim
        self.augment = augment
        self.files = []
        
        for folder in os.listdir(folder_path):
            for file in os.listdir(os.path.join(folder_path, folder)):
                self.files.append(os.path.join(folder_path, folder, file))
        
        if shuffle:
            random.shuffle(self.files)    
        
        if limit_size > 0:
            self.files = self.files[:limit_size]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx): 
        img = read_image(self.files[idx], augment=self.augment, resize_len=[self.dim, self.dim])
        return img