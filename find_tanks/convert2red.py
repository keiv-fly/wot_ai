import cv2
import numpy as np
from matplotlib import pyplot as plt

def test1():
    i1 = img[:, :, 2].copy()
    plt.imshow(i1, cmap='gray')



def red_diff(img):
    r = img[:, :, 2].astype(np.uint16)
    g = img[:, :, 1].astype(np.int32)
    b = img[:, :, 0].astype(np.int32)
    return np.minimum(np.maximum((2 * r - g - b), 0), 255).astype(np.uint8)


img = red_diff(cv2.imread('scenes/scene01801.png',1))
