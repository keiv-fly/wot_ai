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
cv2.im
template = red_diff(cv2.imread('signs/sign_light_red.png',1))
template_mask = cv2.imread('signs/sign_light_red.png',-1)[:,:,3].copy()
template

w, h = template.shape[::-1]
res = cv2.matchTemplate(img,template,cv2.TM_SQDIFF,template_mask)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right, 255, 2)
# plt.subplot(121),plt.imshow(res,cmap = 'gray')
# plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.imshow(res,cmap = 'gray')
plt.imshow(img,cmap = 'gray')

plt.imshow(template_mask,cmap = 'gray')
plt.show()