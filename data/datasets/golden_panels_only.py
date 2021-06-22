import os
import sys
import yaml
import json
from collections import namedtuple
from tqdm import tqdm
import copy
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.config_utils import read_config, Config
from configs.base_config import *
from data.augment import *


class PanelsDataset(Dataset):
    """
    Usage : 
    panels_dataset  = PanelsDataset(images_path = golden_age_config.panel_path,
                         annotation_path = golden_age_config.panels_annotation,
                         panel_dim = golden_age_config.panel_dim ,
                         num_panels = golden_age_config.num_panels,
                         train_test_ratio = golden_age_config.train_test_ratio,
                         normalize = False)
    dataloader = DataLoader(panels_dataset, batch_size=16, shuffle=False,
                num_workers=4)
    
    """

    def __init__(self,
                 images_path: str,
                 annotation_path : str,
                 panel_dim,
                 shuffle: bool = True,
                 augment: bool = False,
                 train_test_ratio: float = 0.95,  # ratio of train data
                 train_mode: bool = True,
                 normalize = False,
                 num_panels = 1,
                 limit_size: int = -1):

        self.images_path = images_path
        self.panel_dim = panel_dim
        self.augment = augment
        self.normalize = normalize
        self.num_panels = num_panels
        
        
        with open(annotation_path) as json_file:
            self.data = json.load(json_file)
        
        train_len = int(len(self.data) * train_test_ratio)

        if train_mode:
            self.data = self.data[:train_len]
        else:
            self.data = self.data[train_len:]

        if shuffle: # Be careful If you want panels to be selected sequentially, you should not shuffle.
            random.shuffle(self.data)
            
        if limit_size != -1:
            self.data = self.data[:limit_size]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        panels = []
        if self.num_panels == 1:
            
            annot = self.data[idx] # Dict with id serie panel information
            panel = Image.open(annot["path"]).convert('RGB')
            panels.append(panel_transforms(panel,self.panel_dim, self.augment))
        else:
            # List of Panels
            annots = self.data[idx: idx +self.num_panels ] # Dict with id serie panel information
            for annot in annots:
                panel = Image.open(annot["path"]).convert('RGB')
                panels.append(panel_transforms(panel, self.panel_dim, self.augment))
            
        
        if self.normalize:
            print("Normalizing", self.normalize)
            panels = normalize(torch.cat(panels, dim=0), means=0.5, stds=0.5)
        else:
            panels = torch.cat(panels, dim=0)
            
        return panels