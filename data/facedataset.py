import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from abc import ABC, abstractmethod
from typing import List, Tuple
from types import SimpleNamespace
from pathlib import Path

# source: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import image_utils
from facedatasource import *

class FaceDataset(Dataset):
    def __init__(self, 
                 datasource: FaceDatasource,
                 transforms=None):
        self.transforms = transforms
        self.datasource = datasource    
                
    def __getitem__(self, index):
        data, face_id = self.datasource.get_item(index)
        if self.transforms is not None:
            data = self.transforms(data)
        # create labels from face_id
        return (img, label)

    def __len__(self):
        return self.datasource.compute_length()

    
def main():
    config = SimpleNamespace()
    config.face_image_folder_path =  "/datasets/iCartoonFace2020/personai_icartoonface_rectrain/icartoonface_rectrain"
    config.max_data_limit = 1000
    ds = iCartoonFaceDatasource(config)

if __name__ == "__main__":
    main()
    
