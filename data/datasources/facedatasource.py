from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
from utils.image_utils import read_image_from_path
from data.datasources.datasource_mode import DataSourceMode
import random
import numpy as np


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
    def get_item_not_belong_to_id(self, not_wanted_id: str) -> FaceDataItem:
        pass

    @abstractmethod
    def data_item_to_actual_data(self, data_item: FaceDataItem):
        pass


class ICartoonFaceDatasource(FaceDatasource):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.data, self.data_by_id = self.load_data()

    def load_data(self):
        if self.mode == DataSourceMode.TRAIN:
            folder_path = self.config.face_image_folder_train_path
        elif self.mode == DataSourceMode.TEST:
            folder_path = self.config.face_image_folder_test_path
        return self.read_face_images(ref_path=folder_path)

    def read_face_images(self, ref_path: str):
        paths = Path(ref_path).glob('**/*.jpg')
        counter = 0
        face_data_items = []
        face_data_items_by_id = {}
        face_data_items_by_id = defaultdict(lambda: [], face_data_items_by_id)
        for path in paths:
            counter += 1
            train_limit = self.config.num_training_samples
            test_min_limit = self.config.test_samples_range[0]
            test_max_limit = self.config.test_samples_range[1]

            if self.mode is DataSourceMode.TRAIN and \
                    train_limit is not None and counter > train_limit:
                break
            elif self.mode is DataSourceMode.TEST and counter < test_min_limit:
                continue
            elif self.mode is DataSourceMode.TEST and counter > test_max_limit:
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
        image = read_image_from_path(data_item.path, self.config.image_dim)
        return image, data_item.face_id

    def get_item_id(self, index: int) -> str:
        face_data_item = self.data[index]
        return face_data_item.face_id

    def get_item_not_belong_to_id(self, not_wanted_id: str) -> FaceDataItem:
        all_keys = self.data_by_id.keys()
        filtered_keys = list(filter(lambda key: key != not_wanted_id, all_keys))
        random_key = random.choice(filtered_keys)
        return random.choice(self.data_by_id[random_key])

    def compute_length(self):
        return len(self.data)
