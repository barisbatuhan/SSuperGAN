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

class GoldenFaceClozeDataset(Dataset):
    
    def __init__(self,
                 images_path :str,
                 panel_annot_path :str, 
                 face_annot_path :str,
                 panel_dim,
                 face_dim :int,
                 num_face_options :int,
                 shuffle :bool=True, 
                 augment :bool=False,
                 random_order :bool=False,
                 mask_val :float=1, # mask with white color for 1 and black color for 0
                 train_test_ratio :float=0.95, # ratio of train data
                 train_mode :bool=True,
                 limit_size :int=-1):
        
        self.images_path = images_path
        self.panel_dim = panel_dim
        self.face_dim = face_dim
        self.augment = augment
        self.mask_val = min(1, max(0, mask_val))
        self.num_face_options = num_face_options
        self.random_order = random_order
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        
        train_series_limit = int(3958 * train_test_ratio) # 3958 is th total number of series
        
        # panel information extraction
        with open(panel_annot_path, "r") as f:
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
            
        # face information extraction
        self.box_annots = {}
        bbox_files = os.listdir(face_annot_path)
        for file in bbox_files:
            # train test split
            
            if file[0] == ".":
                continue
            series_no = int(file.split(".")[0])
            if train_mode and series_no > train_series_limit:
                continue
            elif not train_mode and series_no <= train_series_limit:
                continue
            # reading
            comic_path = os.path.join(face_annot_path, file)
            with open(comic_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if len(line) < 2:
                        continue
                    if line[-1] == "\n":
                        line = line[:-1]
                    
                    parts = line.split(" ")
                    x1, y1, x2, y2 = [int(p) for p in parts[1:5]]
                    conf = float(parts[-1])
                    if conf < 0.9:
                        continue
                    
                    if parts[0] not in self.box_annots.keys():
                        self.box_annots[parts[0]] = []
                    self.box_annots[parts[0]].append([x1, y1, x2, y2])
            
        self.all_files = [*self.box_annots.keys()]
        
        data_len = len(self.data)
        if limit_size > 0:
            self.data = self.data[:min(data_len, limit_size)]
        
        if shuffle:
            random.shuffle(self.data)
            
        
    def __len__(self):
        return len(self.data)
    
    def get_max_iou(self, orig, boxes):
        idx, max_inter = -1, -1
        
        for i, box in enumerate(boxes):
            w = max(0, min(orig[2], box[2]) - max(orig[0], box[0]))
            h = max(0, min(orig[3], box[3]) - max(orig[1], box[1]))
            if max_inter < w*h:
                idx = i
                max_inter = w*h
        
        return idx       
    
    def __getitem__(self, idx): 
        annots = self.data[idx]
        panels, faces = [], []
        
        for i in range(len(annots[0])):
            file_path = os.path.join(self.images_path, annots[0][i])
            file = annots[0][i]
            p_area, f_area = annots[1][i]
            
            panel = Image.open(file_path).convert('RGB')
            w, h = panel.size
            
            if i < len(annots[0]) - 1:
                # earlier panels
                if self.augment:
                    panel = distort_color(panel)
                    panel = horizontal_flip(panel)
                panel = TF.crop(panel, p_area[1], p_area[0], p_area[3]-p_area[1], p_area[2]-p_area[0])
                panel = self.to_tensor(panel).unsqueeze(0)
                panel = TF.resize(panel, [self.panel_dim[1], self.panel_dim[0]])
                panels.append(panel)
            
            else:
                box_areas = self.box_annots[file]
                if len(box_areas) > 1:
                    idx = self.get_max_iou(f_area, box_areas)
                    for box_i, box_area in enumerate(box_areas):
                        # get other faces in the same panel
                        if box_i != idx:
                            crop = [max(0, box_area[0]), max(0, box_area[1]), min(w, box_area[2]), min(h, box_area[3])]
                            fake_face = TF.crop(panel, crop[1], crop[0], crop[3]-crop[1], crop[2]-crop[0])
                            if self.augment:
                                fake_face = distort_color(fake_face)
                                fake_face = horizontal_flip(fake_face)
                            fake_face = self.to_tensor(fake_face).unsqueeze(0)
                            fake_face = TF.resize(fake_face, [self.face_dim, self.face_dim])
                            faces.append(fake_face)
                
                if len(faces) > self.num_face_options - 1:
                    faces = faces[:self.num_face_options-1]
            
                while len(faces) < self.num_face_options - 1:
                    # fill extra slots with random faces
                    rand_file = self.all_files[np.random.randint(0, len(self.all_files))]
                    rand_panel = Image.open(os.path.join(self.images_path, rand_file)).convert('RGB')
                    rw, rh = rand_panel.size
                    rand_annots = self.box_annots[rand_file][0 if np.random.rand() < 0.5 else -1]
                    crop = [max(0, rand_annots[0]), max(0, rand_annots[1]), min(rw, rand_annots[2]), min(rh, rand_annots[3])]
                    fake_face = TF.crop(rand_panel, crop[1], crop[0], crop[3]-crop[1], crop[2]-crop[0])
                    if self.augment:
                        fake_face = distort_color(fake_face)
                        fake_face = horizontal_flip(fake_face)
                    fake_face = self.to_tensor(fake_face)
                    fake_face = fake_face.unsqueeze(0)
                    fake_face = TF.resize(fake_face, [self.face_dim, self.face_dim])
                    faces.append(fake_face)
                
                # get the original face
                f_area = [max(0, f_area[0]), max(0, f_area[1]), min(w, f_area[2]), min(h, f_area[3])]
                orig_face = TF.crop(panel, f_area[1], f_area[0], f_area[3]-f_area[1], f_area[2]-f_area[0])
                if self.augment:
                    orig_face = distort_color(orig_face)
                    orig_face = horizontal_flip(orig_face)
                
                orig_face = self.to_tensor(orig_face)
                orig_face = orig_face.unsqueeze(0)
                orig_face = TF.resize(orig_face, [self.face_dim, self.face_dim])
                faces.append(orig_face)
                
                # get the masked last panel
                panel = self.to_tensor(panel)
                panel = panel.unsqueeze(0)
                panel[:, :, f_area[1]:f_area[3], f_area[0]:f_area[2]] = self.mask_val
                panel = TF.crop(panel, p_area[1], p_area[0], p_area[3]-p_area[1], p_area[2]-p_area[0])
                
                if self.augment:
                    panel = distort_color(panel)
                    panel = horizontal_flip(panel)
                
                panel = TF.resize(panel, [self.panel_dim[1], self.panel_dim[0]])
                panels.append(panel)   
                    
        panels = normalize(torch.cat(panels, dim=0))
              
        faces_tensor = torch.zeros(self.num_face_options, 3, self.face_dim, self.face_dim)
        labels = -1
        
        if not self.random_order:
            faces_order = np.arange(self.num_face_options, dtype=int) 
        else:
            faces_order = np.random.choice(self.num_face_options, size=self.num_face_options, replace=False)
        
        for i, fo in enumerate(faces_order):
            if fo == self.num_face_options-1:
                labels = i
            faces_tensor[i,:,:,:] = faces[fo]
            
        faces_tensor = normalize(faces_tensor)
                
        return panels, faces_tensor, labels 