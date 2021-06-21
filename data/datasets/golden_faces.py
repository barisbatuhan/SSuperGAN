import os
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

from data.augment import read_image

class GoldenFacesDataset(Dataset):
    
    def __init__(self, 
                 folder_path :str, 
                 face_dim :int, 
                 shuffle :bool=False, 
                 train_mode :bool=True, 
                 train_test_ratio :float=0.95, # ratio of train data
                 augment :bool=True, 
                 limit_size :int=-1):
        
        self.dim = face_dim
        self.augment = augment
        self.files = []
        
        train_series_limit = int(3958 * train_test_ratio) # 3958 is th total number of series
        
        for folder in os.listdir(folder_path):
            series_no = int(folder)
            if train_mode and series_no > train_series_limit:
                continue
            elif not train_mode and series_no <= train_series_limit:
                continue
                
            for file in os.listdir(os.path.join(folder_path, folder)):
                self.files.append(os.path.join(folder_path, folder, file))
        
        if shuffle:
            random.shuffle(self.files)    
        
        if limit_size > 0 and limit_size < len(self.files):
            self.files = self.files[:limit_size]
        
    
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, idx): 
        img = read_image(self.files[idx], augment=self.augment, resize_len=[self.dim, self.dim])
        return img