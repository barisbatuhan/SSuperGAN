from matplotlib import pyplot as plt
import numpy as np
from utils import pytorch_util as ptu
from skimage import io, transform
from skimage.color import rgba2rgb, gray2rgb
from skimage.util import crop
from typing import Tuple


def read_image_from_path(path, im_dim: int) -> np.ndarray:
    image = io.imread(path).astype('uint8')
    shape_len = len(image.shape)
    if shape_len < 3:
        image = gray2rgb(image)
    elif shape_len == 3:
        _, _, channels = image.shape
        if channels > 3:
            image = rgba2rgb(image)
    if im_dim is not None:
        return transform.resize(image=image, output_shape=(im_dim, im_dim))
    else:
        return image


def crop_image(whole_image: np.ndarray,
               crop_region: Tuple[int, int, int, int],
               output_shape: Tuple[int, int] = None):
    """
    crops image in numpy formant
    :param whole_image: ndarray
    :param crop_region: y1, x1, y2, x2 (from top-left)
    :param output_shape: if not none image is going to be scaled to this shape
    :return: cropped and (transformed) image
    """
    w, h, _ = whole_image.shape
    w_up_bound = max(w - crop_region[3], 0)
    w_down_bound = max(crop_region[1], 0)
    h_up_bound = max(h - crop_region[2], 0)
    h_down_bound = max(crop_region[0], 0)
    cropped = crop(whole_image,
                   ((w_down_bound, w_up_bound), (h_down_bound, h_up_bound), (0, 0)),
                   copy=False)
    if output_shape is not None:
        return transform.resize(image=cropped, output_shape=output_shape)
    return cropped


def show_ndarray_as_image(array: np.ndarray):
    plt.imshow(array, interpolation='nearest')
    plt.show()


def imshow(img, text=None, should_save=False):
    npimg = ptu.get_numpy(img)
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    test_path = '/home/gsoykan20/Desktop/ffhq_thumbnails/thumbnails128x128/00000.png'
    image = read_image_from_path(test_path, 64)
    show_ndarray_as_image(image)
