import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir

from lib.find import red_diff,get_num_of_enemy

df_files = pd.DataFrame({"name":listdir("scenes")})
df_files["fullname"]= "scenes/" + df_files["name"]
l_matches=[]
for filename in df_files["fullname"]:
    l_matches.append(get_num_of_enemy(filename))

l_matches = list(zip(*l_matches))

df_files["n_light_red"] = l_matches[0]
df_files["n_medium_red"] = l_matches[1]
df_files["red"] = l_matches[2]
df_files["white"] = l_matches[3]
df_files["black"] = l_matches[4]
df_files["red_bw"] = l_matches[5]

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 180)
df_files


threshold=0.20

img = red_diff(cv2.imread('scenes/scene02701.png',1))
_ = cv2.rectangle(img,(0,720),(368,1090),(0,0,0),-1)
_ = cv2.rectangle(img,(1000,0),(1228,36),(0,0,0),-1)
template = red_diff(cv2.imread('signs/sign_light_red.png',1))

w, h = template.shape[::-1]
res = cv2.matchTemplate(img,template,cv2.TM_SQDIFF_NORMED)
#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
loc = np.where( res <= threshold)

if loc[0].shape[0] >0:
    loc = list(zip(*loc[::-1]))
    loc = np.array(loc)
    loc0 = loc[0]
    mask = np.ones(loc.shape[0], dtype=np.bool)
    for i in range(loc.shape[0]):
        mask = mask & (np.sum(loc - loc[i], axis=1) > 20)

    mask[0] = True
    loc = loc[mask]
else:
    loc=[]

print("Matches:",loc.shape[0])
for pt in loc:
    _ = cv2.rectangle(img, tuple(pt), (pt[0] + w, pt[1] + h), 255, 2)
plt.imshow(img,cmap = 'gray')

plt.imshow(res,cmap = 'gray')
plt.show()