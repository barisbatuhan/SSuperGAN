from matplotlib import pyplot as plt
import numpy as np

def show_ndarray_as_image(array: np.ndarray):
    plt.imshow(array, interpolation='nearest')
    plt.show()