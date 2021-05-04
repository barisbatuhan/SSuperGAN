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
        face_data = np.empty((0, self.config.image_dim, self.config.image_dim, 3))
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
            image = read_image_from_path(global_path, self.config.image_dim)
            # TODO: get rid of concat
            face_data = np.concatenate((face_data, np.expand_dims(image, axis=0)), axis=0)

            if face_data.shape[0] % 512 == 0:
                print("reading image: " + str(face_data.shape[0]))

        return face_data

    def compute_length(self) -> int:
        return self.data.shape[0]

    def get_item(self, index: int):
        return self.data[index, :, :, :]
