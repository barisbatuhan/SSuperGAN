from pathlib import Path
from utils.image_utils import read_image_from_path
import numpy as np
from data.datasources.base_datasource import BaseDatasource
from data.datasources.datasource_mode import DataSourceMode


# Dataset Source: https://github.com/NVlabs/ffhq-dataset
# https://drive.google.com/drive/folders/1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv
class FFHQDatasource(BaseDatasource):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.data = self.load_data()

    def load_data(self):
        if self.mode == DataSourceMode.TRAIN:
            folder_path = self.config.face_image_folder_train_path
        elif self.mode == DataSourceMode.TEST:
            folder_path = self.config.face_image_folder_test_path
        else:
            raise NotImplementedError
        return self.read_face_images(ref_path=folder_path)

    def read_face_images(self, ref_path: str):
        paths = Path(ref_path).glob('**/*.png')
        counter = 0
        added_image_counter = 0

        train_limit = self.config.num_training_samples
        test_min_limit = self.config.test_samples_range[0]
        test_max_limit = self.config.test_samples_range[1]

        if self.mode is DataSourceMode.TRAIN:
            total_image_count = train_limit
        elif self.mode is DataSourceMode.TEST and counter < test_min_limit:
            total_image_count = test_max_limit - test_min_limit

        face_data = np.empty((total_image_count, self.config.image_dim, self.config.image_dim, 3))

        for path in paths:
            counter += 1

            if self.mode is DataSourceMode.TRAIN and \
                    train_limit is not None and counter > train_limit:
                break
            elif self.mode is DataSourceMode.TEST and counter < test_min_limit:
                continue
            elif self.mode is DataSourceMode.TEST and counter > test_max_limit:
                break

            global_path = str(path)
            image = read_image_from_path(global_path, self.config.image_dim)

            if self.mode is DataSourceMode.TRAIN:
                face_data[counter - 1, :, :, :] = image
            elif self.mode is DataSourceMode.TEST:
                face_data[counter - 1 - test_min_limit, :, :, :] = image

            added_image_counter += 1

            if added_image_counter % 512 == 0:
                print("reading image: " + str(added_image_counter))

        return face_data

    def compute_length(self) -> int:
        return self.data.shape[0]

    def get_item(self, index: int):
        return self.data[index, :, :, :]
