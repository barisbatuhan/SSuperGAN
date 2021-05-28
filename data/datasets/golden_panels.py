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

sort_golden_panel_dataset = True


class GoldenPanelsDataset(Dataset):

    def __init__(self,
                 images_path: str,
                 annot_path: str,
                 panel_dim,
                 face_dim: int,
                 shuffle: bool = True,
                 augment: bool = True,
                 mask_val: float = 1,  # mask with white color for 1 and black color for 0
                 mask_all: bool = False,  # masks faces from all panels and returns all faces
                 return_mask: bool = False,  # returns last panel's masking information
                 return_mask_coordinates=False,  # returns coordinates of the masked area
                 train_test_ratio: float = 0.95,  # ratio of train data
                 train_mode: bool = True,
                 limit_size: int = -1):

        self.images_path = images_path
        self.panel_dim = panel_dim
        self.face_dim = face_dim
        self.augment = augment
        self.mask_val = min(1, max(0, mask_val))
        self.mask_all = mask_all
        self.return_mask = return_mask
        self.return_mask_coordinates = return_mask_coordinates

        with open(annot_path, "r") as f:
            annots = json.load(f)

        #### Sorting By Comic Number
        if sort_golden_panel_dataset:
            def sort_by_comic_number(element):
                return int(element[0][0].split("/")[0])

            annots_values = list(annots.values())
            annots_values.sort(key=sort_by_comic_number)

            for i in range(len(annots_values)):
                annots[str(i)] = annots_values[i]

        #### Sorting By Comic Number

        self.data = []
        for k in annots.keys():
            if 0 < limit_size < int(k):
                break
            else:
                self.data.append(annots[k])

        train_len = int(len(self.data) * train_test_ratio)

        if train_mode:
            self.data = self.data[:train_len]
        else:
            self.data = self.data[train_len:]

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
                f_shifted[fi] = f_area[fi] - p_area[fi % 2]

            panel = Image.open(file).convert('RGB')
            if self.augment:
                panel = distort_color(panel)

            panel = TF.crop(panel, p_area[1], p_area[0], p_area[3] - p_area[1], p_area[2] - p_area[0])
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
                    m_shifted = [  # after resize, these dims will be masked
                        min(int(round(f_shifted[0] * self.panel_dim[0] / W)), self.panel_dim[0]),
                        min(int(round(f_shifted[1] * self.panel_dim[1] / H)), self.panel_dim[1]),
                        min(int(round(f_shifted[2] * self.panel_dim[0] / W)), self.panel_dim[0]),
                        min(int(round(f_shifted[3] * self.panel_dim[1] / H)), self.panel_dim[1]),
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

        if self.return_mask and self.return_mask_coordinates:
            return panels, faces, mask_data, mask_coordinates
        if self.return_mask:
            return panels, faces, mask_data
        else:
            return panels, faces
