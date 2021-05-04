from matplotlib import pyplot as plt
import numpy as np
from utils import pytorch_util as ptu
from skimage import io, transform
from skimage.color import rgba2rgb, gray2rgb


def read_image_from_path(path, im_dim: int) -> np.ndarray:
    image = io.imread(path).astype('uint8')
    shape_len = len(image.shape)
    if shape_len < 3:
        image = gray2rgb(image)
    elif shape_len == 3:
        _, _, channels = image.shape
        if channels > 3:
            image = rgba2rgb(image)
    image = transform.resize(image=image, output_shape=(im_dim, im_dim))
    return image


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
