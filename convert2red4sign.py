import cv2

from lib.find_lib import red_diff

img = red_diff(cv2.imread('scenes/scene01801.png',1))
cv2.imwrite("transformed",img)
