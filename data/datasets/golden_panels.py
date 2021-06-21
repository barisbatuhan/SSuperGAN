import os
import json
import copy
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from data.augment import *

class GoldenPanelsDataset(Dataset):
    
    def __init__(self,
                 images_path :str,
                 annot_path :str, 
                 panel_dim,
                 face_dim :int, 
                 shuffle :bool=True, 
                 augment :bool=True, 
                 mask_val :float=1, # mask with white color for 1 and black color for 0
                 mask_all :bool=False, # masks faces from all panels and returns all faces
                 return_mask :bool=False, # returns last panel's masking information
                 return_mask_coordinates :bool=False,
                 train_test_ratio :float=0.95, # ratio of train data
                 train_mode :bool=True,
                 limit_size :int=-1):
        
        self.images_path = images_path
        self.panel_dim = panel_dim
        self.face_dim = face_dim
        self.augment = augment
        self.mask_val = min(1, max(0, mask_val))
        self.mask_all = mask_all
        self.return_mask = return_mask
        self.return_mask_coordinates = return_mask_coordinates
        
        
        train_series_limit = int(3958 * train_test_ratio) # 3958 is th total number of series
        
        # panel information extraction
        with open(annot_path, "r") as f:
            annots = json.load(f)

        self.data = []
        for k in annots.keys():
            # train test split
            series_no = int(annots[k][0][0].split("/")[0])
            if train_mode and series_no > train_series_limit:
                continue
            elif not train_mode and series_no <= train_series_limit:
                continue
            # data reading
            self.data.append(annots[k])  
               
        data_len = len(self.data)
        if limit_size > 0:
            self.data = self.data[:min(data_len, limit_size)]
        
        if shuffle:
            random.shuffle(self.data)
            
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        annots = self.data[idx]
        panels, faces = [], []
        
        if self.return_mask:
            mask_data = torch.zeros(self.panel_dim[1], self.panel_dim[0])
        
        for i in range(len(annots[0])):
            file = os.path.join(self.images_path, annots[0][i])
            p_area, f_area = annots[1][i]
            
            f_shifted = [-1, -1, -1, -1]
            for fi in range(len(f_area)):
                f_shifted[fi] = f_area[fi] - p_area[fi%2]
            
            panel = Image.open(file).convert('RGB')
            if self.augment:
                panel = distort_color(panel)
            
            panel = TF.crop(panel, p_area[1], p_area[0], p_area[3]-p_area[1], p_area[2]-p_area[0])
            panel = transforms.ToTensor()(panel).unsqueeze(0)
            
            if self.mask_all or i == len(annots[0]) - 1:
                # get face
                face = copy.deepcopy(panel[:, :, f_shifted[1]:f_shifted[3], f_shifted[0]:f_shifted[2]])
                face = TF.resize(face, [self.face_dim, self.face_dim])
                if not self.mask_all:
                    face = face.squeeze(0)
                faces.append(face)
                # mask panel
                panel[:, :, f_shifted[1]:f_shifted[3], f_shifted[0]:f_shifted[2]] = self.mask_val
                
                if self.return_mask and i == len(annots[0]) - 1:
                    _, _, H, W = panel.shape
                    m_shifted = [ # after resize, these dims will be masked
                        int(math.floor(f_shifted[0] * self.panel_dim[0] / W)),
                        int(math.floor(f_shifted[1] * self.panel_dim[1] / H)),
                        int(math.ceil(f_shifted[2] * self.panel_dim[0] / W)),
                        int(math.ceil(f_shifted[3] * self.panel_dim[1] / H)),
                    ]
                    mask_data[m_shifted[1]:m_shifted[3], m_shifted[0]:m_shifted[2]] = 1
                    mask_coordinates = np.array((m_shifted[1], m_shifted[3], m_shifted[0], m_shifted[2]))
                
            
            panel = TF.resize(panel, [self.panel_dim[1], self.panel_dim[0]])
            panels.append(panel)      
        
        panels = normalize(torch.cat(panels, dim=0))
        if self.mask_all:
            faces = normalize(torch.cat(faces, dim=0))
        else:
            faces = normalize(faces[0])

        if self.return_mask:
            if self.return_mask_coordinates:
                return panels, faces, mask_data, mask_coordinates
            else:
                return panels, faces, mask_data
        else:
            if self.return_mask_coordinates:
                return panels, faces, mask_coordinates
            else:
                return panels, faces