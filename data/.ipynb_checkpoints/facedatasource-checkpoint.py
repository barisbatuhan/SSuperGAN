import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple
from types import SimpleNamespace
# source: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import image_utils

class FaceDataItem:
    def __init__(self, 
                 path,
                 face_id):
        self.path = path
        self.face_id = face_id

class FaceDatasource(ABC):
    def __init__(self, config):
        self.config = config
        
    @abstractmethod
    def load_data(self) -> List[FaceDataItem]:
        pass
        
    @abstractmethod
    def compute_length(self) -> int:
        pass
    
    # returns image and its face id
    @abstractmethod
    def get_item(self, index: int) -> Tuple[np.ndarray, str]:
        pass
    
    
class iCartoonFaceDatasource(FaceDatasource):
    def __init__(self, config):
        super().__init__(config)
        self.data = self.load_data()
    
    def load_data(self):
        folder_path = self.config.face_image_folder_path
        return self.read_face_images(ref_path=folder_path, limit=self.config.max_data_limit)
    
    def read_face_images(self, ref_path: str, limit: int):
        paths = Path(ref_path).glob('**/*.jpg')
        counter = 0
        face_data_items = []
        for path in paths:
            counter += 1
            if limit is not None and counter > limit:
                break
            global_path = str(path)
            face_tag = str(path.parts[-2])
            face_data_items.append(FaceDataItem(global_path, face_id=face_tag))
        return face_data_items
    
    def get_item(self, index):
        face_data_item = self.data[index]
        image = io.imread(face_data_item.path)
        return (image, face_data_item.face_id)
    
    def compute_length(self):
        return len(self.data)     