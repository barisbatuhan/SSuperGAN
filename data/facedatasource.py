import numpy as np
from skimage import io, transform
from PIL import Image
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple
from types import SimpleNamespace
#  source: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
import sys, os, inspect
from collections import defaultdict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data.datasource_mode import DataSourceMode
from utils import image_utils
import random


class FaceDataItem:
    def __init__(self,
                 path,
                 face_id):
        self.path = path
        self.face_id = face_id


class FaceDatasource(ABC):
    def __init__(self, config, mode: DataSourceMode):
        self.config = config
        self.mode = mode
        self.data = None
        self.data_by_id = None

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

    @abstractmethod
    def get_item_id(self, index: int) -> str:
        pass
    
    @abstractmethod
    def get_item_not_belong_to_id(self, id: str) -> FaceDataItem:
        pass

    @abstractmethod
    def data_item_to_actual_data(self, data_item: FaceDataItem):
        pass


class ICartoonFaceDatasource(FaceDatasource):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.data, self.data_by_id = self.load_data()

    def load_data(self):
        folder_path = self.config.face_image_folder_path
        return self.read_face_images(ref_path=folder_path)

    # TODO: TRAIN - TEST SPLIT LOGIC MIGHT BE MOVED TO SOMEWHERE ELSE
    def read_face_images(self, ref_path: str):
        paths = Path(ref_path).glob('**/*.jpg')
        counter = 0
        face_data_items = []
        face_data_items_by_id = {}
        face_data_items_by_id = defaultdict(lambda: [], face_data_items_by_id)
        for path in paths:
            counter += 1
            train_limit = self.config.num_training_samples
            test_limit = self.config.num_training_samples + self.config.num_test_samples

            if self.mode is DataSourceMode.TRAIN and \
                    train_limit is not None and counter > train_limit:
                break
            elif self.mode is DataSourceMode.TEST and counter < train_limit:
                continue
            elif self.mode is DataSourceMode.TEST and counter > test_limit:
                break

            global_path = str(path)
            face_tag = str(path.parts[-2])
            face_data_item = FaceDataItem(global_path, face_id=face_tag)
            face_data_items.append(face_data_item)
            face_data_items_by_id[face_tag].append(face_data_item)
        return face_data_items, face_data_items_by_id

    def get_item(self, index):
        face_data_item = self.data[index]
        return self.data_item_to_actual_data(face_data_item)

    def data_item_to_actual_data(self, data_item: FaceDataItem):
        image = io.imread(data_item.path).astype('uint8')
        im_dim = self.config.image_dim
        image = transform.resize(image=image, output_shape=(im_dim, im_dim))
        return image, data_item.face_id

    def get_item_id(self, index: int) -> str:
        face_data_item = self.data[index]
        return face_data_item.face_id
    
    def get_item_not_belong_to_id(self, id: str) -> FaceDataItem:
        # TODO: implement
        random.choice(self.datasource.data_by_id[img0_id])

    def compute_length(self):
        return len(self.data)
