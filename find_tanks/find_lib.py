import numpy as np

def red_diff(img):
    r = img[:, :, 2].astype(np.uint16)
    g = img[:, :, 1].astype(np.int32)
    b = img[:, :, 0].astype(np.int32)
    return np.minimum(np.maximum((2 * r - g - b), 0), 255).astype(np.uint8)