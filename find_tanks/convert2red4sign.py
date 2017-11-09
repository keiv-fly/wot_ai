import cv2
import numpy as np

import find_lib


img = find_lib.red_diff(cv2.imread('scenes/scene01801.png',1))
