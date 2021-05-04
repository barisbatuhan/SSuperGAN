from pathlib import Path

from utils.config_utils import read_config, Config
from utils.image_utils import read_image_from_path, crop_image, show_ndarray_as_image
import numpy as np
from data.datasources.base_datasource import BaseDatasource
from data.datasources.datasource_mode import DataSourceMode


# Dataset Source: https://github.com/NVlabs/ffhq-dataset
# https://drive.google.com/drive/folders/1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv
class GoldenAgeFaceDatasource(BaseDatasource):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.data = self.load_data()

    def load_data(self):
        annotations_folder_path = self.config.annotations_folder_path
        return self.read_face_images(annot_path=annotations_folder_path)

    def read_face_images(self, annot_path: str):
        if self.mode is DataSourceMode.TRAIN:
            annotation_paths = Path(annot_path).glob('**/*.txt')
            total_image_count = self.config.num_training_samples
        elif self.mode is DataSourceMode.TEST:
            annotation_paths = reversed(list(Path(annot_path).glob('**/*.txt')))
            total_image_count = self.config.num_test_samples
        else:
            raise NotImplementedError

        added_image_counter = 0

        face_data = np.empty((total_image_count, self.config.image_dim, self.config.image_dim, 3))

        for annot_path in annotation_paths:
            annot_path_str = str(annot_path)
            with open(annot_path_str) as fp:
                for line in fp:
                    image_location, y1, x1, y2, x2, confidence = line.strip().split()
                    y1, x1, y2, x2 = list(map(lambda x: int(x), [y1, x1, y2, x2]))
                    y_dim = abs(y1 - y2)
                    x_dim = abs(x1 - x2)
                    min_dim = min(x_dim, y_dim)
                    if min_dim < self.config.min_original_face_dim:
                        continue
                    image_location = self.config.panel_folder_path + '/' + image_location
                    whole_image = read_image_from_path(image_location, im_dim=None)
                    cropped_face = crop_image(whole_image,
                                              crop_region=(y1, x1, y2, x2),
                                              output_shape=(self.config.image_dim, self.config.image_dim))
                    face_data[added_image_counter, :, :, :] = cropped_face
                    added_image_counter += 1
                    if added_image_counter >= total_image_count:
                        return face_data

        return face_data

    def compute_length(self) -> int:
        return self.data.shape[0]

    def get_item(self, index: int):
        return self.data[index, :, :, :]


if __name__ == '__main__':
    config = read_config(Config.GOLDEN_AGE_FACE)
    datasource = GoldenAgeFaceDatasource(config, mode=DataSourceMode.TRAIN)
