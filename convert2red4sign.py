import cv2

from lib.find import red_diff

img = red_diff(cv2.imread('scenes/scene03751.png',1))
_ = cv2.imwrite("transformed_data/sc2.png",img)

# convert to grey
img = cv2.imread('scenes/scene02401.png',1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_ = cv2.imwrite("transformed_data/sc1Grey.png",img)