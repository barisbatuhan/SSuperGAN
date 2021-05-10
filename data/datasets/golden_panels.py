import os
import json
import copy
import random
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
                 limit_size :int=-1):
        
        self.images_path = images_path
        self.panel_dim = panel_dim
        self.face_dim = face_dim
        self.augment = augment
        self.mask_val = min(1, max(0, mask_val))
        self.mask_all = mask_all
        
        with open(annot_path, "r") as f:
            annots = json.load(f) 
        
        self.data = []
        for k in annots.keys():
            if limit_size > 0 and k > limit_size:
                break
            else:
                self.data.append(annots[k])
        
        if shuffle:
            random.shuffle(self.data)    
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        annots = self.data[idx]
        panels, faces = [], []
        
        for i in range(len(annots[0])):
            file = os.path.join(self.images_path, annots[0][i])
            p_area, f_area = annots[1][i]
            
            for fi in range(len(f_area)):
                f_area[fi] -= p_area[fi%2]
            
            panel = Image.open(file).convert('RGB')
            if self.augment:
                panel = distort_color(img)
            
            panel = TF.crop(panel, p_area[1], p_area[0], p_area[3]-p_area[1], p_area[2]-p_area[0])
            panel = transforms.ToTensor()(panel).unsqueeze(0)
            
            if self.mask_all or i == len(annots[0]) - 1:
                # get face
                face = copy.deepcopy(panel[:, :, f_area[1]:f_area[3], f_area[0]:f_area[2]])
                face = TF.resize(face, [self.face_dim, self.face_dim])
                if not self.mask_all:
                    face = face.squeeze(0)
                faces.append(face)
                # mask panel
                panel[:, :, f_area[1]:f_area[3], f_area[0]:f_area[2]] = self.mask_val
            
            panel = TF.resize(panel, [self.panel_dim[1], self.panel_dim[0]])
            panels.append(panel)      
        
        panels = normalize(torch.cat(panels, dim=0))
        if self.mask_all:
            faces = normalize(torch.cat(faces, dim=0))
        else:
            faces = normalize(faces[0])

        return panels, faces