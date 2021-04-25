from matplotlib import pyplot as plt
import numpy as np
from utils import pytorch_util as ptu


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
