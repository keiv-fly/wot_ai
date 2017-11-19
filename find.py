import cv2
import multiprocessing
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir
from line_profiler import LineProfiler
from multiprocessing.dummy import Pool as ThreadPool
from lib.find import red_diff,get_num_of_enemy, get_num_of_enemy_parallel, get_num_of_enemy_internal_parallel

df_files = pd.DataFrame({"name":listdir("scenes")})
df_files["fullname"]= "scenes/" + df_files["name"]
l_imgs=[]
for filename in df_files["fullname"]:
    l_imgs.append(cv2.imread(filename, 1))


# l_matches = []
# for img0 in l_imgs:
#     l_matches.append(get_num_of_enemy(img0))

# 18.2s per 61 images = 3.35 fps for all types of matching TM_SQDIFF_NORMED
# 9.55s per 61 images = 6.38 fps for 2 I templates TM_SQDIFF_NORMED
# 8.13s TM_CCOEFF correctness of data unknown
# 8.79 TM_CCOEFF_NORMED data proved
# 54.4s TM_CCORR only 5% is matching the other time is filtering
# 55.6s TM_CCORR with numba jit wo types
# 6.25s TM_CCORR_NORMED bad results
# 8.27s, 9.13s TM_SQDIFF_NORMED with numba jit (6.7-7.4 fps)
# 52.1s per 61 images TM_SQDIFF_NORMED with numba jit, parallel with multiprocess
# 10.2s with "with multiprocessing.Pool(processes=2) here"
# 9.87s with measuring after "pool = multiprocessing.Pool(processes=2)"
# 10.0s with processes=4
# 12.2s with processes=1
# 6.62s with threads=2
# 8.41s with threads=1
# 6.73s with threads=4
# 9.45s with internal pool creation

#pool = multiprocessing.Pool(processes=1)
#pool = ThreadPool(4)

l_matches = []
for img0 in l_imgs:
    l_matches.append(get_num_of_enemy_internal_parallel(img0))

# l_matches = []
# for img0 in l_imgs:
#     l_matches.append(get_num_of_enemy_parallel(img0,pool))



#%%time
# with multiprocessing.Pool(processes=2) as pool:
#     for img0 in l_imgs:
#         l_matches.append(get_num_of_enemy_parallel(img0,pool))


lp = LineProfiler()
lp_wrapper = lp(get_num_of_enemy)
lp_wrapper(l_imgs[0])
lp.print_stats()


l_matches = list(zip(*l_matches))

#df_files["n_light_red"] = l_matches[0]
#df_files["n_medium_red"] = l_matches[1]
df_files["red"] = l_matches[0]
df_files["white"] = l_matches[1]
df_files["black"] = l_matches[2]
df_files["red_bw"] = l_matches[3]
df_files["red_other"] = l_matches[4]

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 180)
df_files

df_files.to_csv("tests/match.csv", index=False)


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