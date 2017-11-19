import numpy as np
import cv2
from itertools import repeat
import numba
from multiprocessing.dummy import Pool as ThreadPool

#np.core.arrayprint._line_width = 180

def remove_close(loc):
    loc = np.array(loc)
    loc_filtered = []

    n = loc.shape[0]
    for i in range(n):
        loc0 = loc[0]
        loc_filtered.append(loc0)
        loc = loc[1:]
        loc = loc[np.abs(np.sum(loc - loc0[np.newaxis,:], axis=1))>6]
        if len(loc)==0:
            break
    return loc_filtered

remove_close_jit = numba.jit(remove_close)


def red_diff(img):
    r = img[:, :, 2].astype(np.uint16)
    g = img[:, :, 1].astype(np.int32)
    b = img[:, :, 0].astype(np.int32)
    return np.minimum(np.maximum((2 * r - g - b), 0), 255).astype(np.uint8)



filename='scenes/scene01051.png'
filename='scenes/102.png'
filename='scenes/scene03151.png'
filename='scenes/scene08251.png'
def get_num_of_enemy(img0):
    threshold = 0.20
    #
    # img = red_diff(img0)
    # _ = cv2.rectangle(img, (0, 720), (368, 1090), (0, 0, 0), -1)
    # _ = cv2.rectangle(img, (1000, 0), (1228, 36), (0, 0, 0), -1)
    #
    # # light red
    # template = cv2.imread('signs/sign_light_red_augm.png', 0)
    #
    # w, h = template.shape[::-1]
    # res_match = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    # # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # loc = np.where(res_match <= threshold)
    #
    # if loc[0].shape[0] > 0:
    #     loc = list(zip(*loc[::-1]))
    #     loc = np.array(loc)
    #     loc0 = loc[0]
    #     mask = np.ones(loc.shape[0],dtype=np.bool)
    #     for i in range(loc.shape[0]):
    #         mask = mask & (np.sum(loc - loc[i], axis=1) > 20)
    #
    #     mask[0] = True
    #     loc = loc[mask]
    # else:
    #     loc = np.array([])
    #
    # light_red = loc.shape[0]
    #
    # # med red
    # template = cv2.imread('signs/sign_med_red_augm.png', 0)
    #
    # w, h = template.shape[::-1]
    # res_match = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    # # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # loc = np.where(res_match <= threshold)
    #
    # if loc[0].shape[0] > 0:
    #     loc = list(zip(*loc[::-1]))
    #     loc = np.array(loc)
    #     loc0 = loc[0]
    #     mask = np.ones(loc.shape[0],dtype=np.bool)
    #     for i in range(loc.shape[0]):
    #         mask = mask & (np.sum(loc - loc[i], axis=1) > 20)
    #
    #     mask[0] = True
    #     loc = loc[mask]
    # else:
    #     loc = np.array([])
    #
    # medium_red = loc.shape[0]

    #class1
    threshold = 0.10
    template1 = cv2.imread('signs/sign_I_2.png', 0)
    template2 = cv2.imread('signs/sign_I_3.png', 0)

    w, h = template1.shape[::-1]
    img_grey = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    _ = cv2.rectangle(img_grey, (0, 720), (368, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (690, 0), (1228, 36), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (0, 0), (68, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1854, 0), (1920, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1800, 1000), (1920, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (750, 1048), (1385, 1090), (0, 0, 0), -1)


    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # p1 = Process(target=one_match, args=(template1,img_grey,return_dict,1))
    # p2 = Process(target=one_match, args=(template2, img_grey, return_dict, 2))
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
    # vals = return_dict.values()
    # res_match1 = vals[0]
    # res_match2 = vals[1]

    res_match1 = cv2.matchTemplate(img_grey, template1, cv2.TM_SQDIFF_NORMED)
    res_match2 = cv2.matchTemplate(img_grey, template2, cv2.TM_SQDIFF_NORMED)
    loc1 = np.where(res_match1 <= threshold)
    loc2 = np.where(res_match2 <= threshold)
    loc1 = list(zip(*loc1[::-1]))
    loc2 = list(zip(*loc2[::-1]))
    loc=loc1+loc2
    loc = list(set(loc))


    if len(loc) > 0:
        loc = remove_close_jit(loc)
        # loc = np.array(loc)
        # loc_filtered = []
        #
        # n = loc.shape[0]
        # for i in range(n):
        #     loc0 = loc[0]
        #     loc_filtered.append(loc0)
        #     loc = loc[1:]
        #     loc = loc[np.abs(np.sum(loc - loc0[np.newaxis,:], axis=1))>6]
        #     if len(loc)==0:
        #         break
        # loc = loc_filtered
    else:
        loc = np.array([])

    #loc[0]+[w/2,h/2]

    loc_npa=np.array(loc)

    # img0[:,:,2][loc_npa[0,0]:(loc_npa[0,0]+w),(loc_npa[0,1]+73):(loc_npa[0,1]+95)]
    # yrange1=np.arange(73,73+4)
    # yrange2=np.arange(73+10,95)
    # xrange=np.arange(0,w)

    loc_red = []
    loc_white = []
    loc_black = []
    loc_red_bw = []
    loc_red_other = []

    if len(loc) > 0:
        colors_list = []

        i=0
        for i in range(len(loc)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]

            if len(t1)==0:
                colors_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g=np.median(
                    img0[:,:,1][(loc_npa[i,1]+73):(loc_npa[i,1]+91),loc_npa[i,0]:(loc_npa[i,0]+w)]
            )
            b=np.median(
                    img0[:,:,0][(loc_npa[i,1]+73):(loc_npa[i,1]+91),loc_npa[i,0]:(loc_npa[i,0]+w)]
            )
            # b=np.median(
            #     np.concatenate([
            #         img0[:,:,0][loc_npa[i,0]:(loc_npa[i,0]+w),(loc_npa[i,1]+73):(loc_npa[i,1]+73+4)],
            #         img0[:, :, 0][loc_npa[i, 0]:(loc_npa[i, 0] + w), (loc_npa[i,1]+73+10):(loc_npa[i, 1]+95)]
            #     ], axis=1)
            # )
            colors_list.append((r,g,b))
        good_colors = np.array(((200, 0, 10),(229,229,226)))

        red = (200, 0, 10)
        t1=np.array(colors_list) - red
        mask1 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_red = loc_npa[mask1]

        white = (229,229,226)
        t1=np.array(colors_list) - white
        mask2 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_white = loc_npa[mask2]

        black = (4,2,3)
        t1=np.array(colors_list) - black
        mask3 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_black = loc_npa[mask3]

        other_npa = loc_npa[~mask1&~mask2&~mask3].copy()

        b_w = np.concatenate((loc_white,loc_black))
        colors_bw_list = []

        loc_npa = b_w
        i=2
        for i in range(len(b_w)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size==0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g=np.median(
                img0[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b=np.median(
                img0[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            colors_bw_list.append((r,g,b))
        if len(colors_bw_list)>0:
            red = (200, 0, 10)
            t1 = np.array(colors_bw_list) - red
            mask = ~np.isnan(t1).any(axis=1)
            loc_npa = loc_npa[mask]
            t1=t1[mask]
            loc_red_bw = loc_npa[np.sum(np.power(t1,2),axis=1)<3000]

        colors_other_list = []
        loc_npa = other_npa
        i=0
        for i in range(len(other_npa)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size==0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g=np.median(
                img0[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b=np.median(
                img0[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            colors_other_list.append((r,g,b))
        if len(colors_other_list)>0:
            red = (200, 0, 10)
            t1 = np.array(colors_other_list) - red
            mask = ~np.isnan(t1).any(axis=1)
            loc_npa = loc_npa[mask]
            t1=t1[mask]
            loc_red_other = loc_npa[np.sum(np.power(t1,2),axis=1)<3000]

    return len(loc_red), len(loc_white), len(loc_black), len(loc_red_bw), len(loc_red_other)

def get_num_of_enemy_parallel(img0,pool):
    threshold = 0.10
    template1 = cv2.imread('signs/sign_I_2.png', 0)
    template2 = cv2.imread('signs/sign_I_3.png', 0)

    w, h = template1.shape[::-1]
    img_grey = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    _ = cv2.rectangle(img_grey, (0, 720), (368, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (690, 0), (1228, 36), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (0, 0), (68, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1854, 0), (1920, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1800, 1000), (1920, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (750, 1048), (1385, 1090), (0, 0, 0), -1)

    templates = (template1,template2)
    res_matchs = pool.starmap(cv2.matchTemplate, zip(repeat(img_grey),templates,repeat(cv2.TM_SQDIFF_NORMED)))
    #res_match1 = cv2.matchTemplate(img_grey, template1, cv2.TM_SQDIFF_NORMED)
    #res_match2 = cv2.matchTemplate(img_grey, template2, cv2.TM_SQDIFF_NORMED)
    locs = (np.where(res_match <= threshold) for res_match in res_matchs)
    loc1, loc2 = (list(zip(*x[::-1])) for x in locs)

    loc=loc1+loc2
    loc = list(set(loc))


    if len(loc) > 0:
        loc = remove_close_jit(loc)
        # loc = np.array(loc)
        # loc_filtered = []
        #
        # n = loc.shape[0]
        # for i in range(n):
        #     loc0 = loc[0]
        #     loc_filtered.append(loc0)
        #     loc = loc[1:]
        #     loc = loc[np.abs(np.sum(loc - loc0[np.newaxis,:], axis=1))>6]
        #     if len(loc)==0:
        #         break
        # loc = loc_filtered
    else:
        loc = np.array([])

    #loc[0]+[w/2,h/2]

    loc_npa=np.array(loc)

    # img0[:,:,2][loc_npa[0,0]:(loc_npa[0,0]+w),(loc_npa[0,1]+73):(loc_npa[0,1]+95)]
    # yrange1=np.arange(73,73+4)
    # yrange2=np.arange(73+10,95)
    # xrange=np.arange(0,w)

    loc_red = []
    loc_white = []
    loc_black = []
    loc_red_bw = []
    loc_red_other = []

    if len(loc) > 0:
        colors_list = []

        i=0
        for i in range(len(loc)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]

            if len(t1)==0:
                colors_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g=np.median(
                    img0[:,:,1][(loc_npa[i,1]+73):(loc_npa[i,1]+91),loc_npa[i,0]:(loc_npa[i,0]+w)]
            )
            b=np.median(
                    img0[:,:,0][(loc_npa[i,1]+73):(loc_npa[i,1]+91),loc_npa[i,0]:(loc_npa[i,0]+w)]
            )
            # b=np.median(
            #     np.concatenate([
            #         img0[:,:,0][loc_npa[i,0]:(loc_npa[i,0]+w),(loc_npa[i,1]+73):(loc_npa[i,1]+73+4)],
            #         img0[:, :, 0][loc_npa[i, 0]:(loc_npa[i, 0] + w), (loc_npa[i,1]+73+10):(loc_npa[i, 1]+95)]
            #     ], axis=1)
            # )
            colors_list.append((r,g,b))
        good_colors = np.array(((200, 0, 10),(229,229,226)))

        red = (200, 0, 10)
        t1=np.array(colors_list) - red
        mask1 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_red = loc_npa[mask1]

        white = (229,229,226)
        t1=np.array(colors_list) - white
        mask2 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_white = loc_npa[mask2]

        black = (4,2,3)
        t1=np.array(colors_list) - black
        mask3 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_black = loc_npa[mask3]

        other_npa = loc_npa[~mask1&~mask2&~mask3].copy()

        b_w = np.concatenate((loc_white,loc_black))
        colors_bw_list = []

        loc_npa = b_w
        i=2
        for i in range(len(b_w)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size==0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g=np.median(
                img0[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b=np.median(
                img0[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            colors_bw_list.append((r,g,b))
        if len(colors_bw_list)>0:
            red = (200, 0, 10)
            t1 = np.array(colors_bw_list) - red
            mask = ~np.isnan(t1).any(axis=1)
            loc_npa = loc_npa[mask]
            t1=t1[mask]
            loc_red_bw = loc_npa[np.sum(np.power(t1,2),axis=1)<3000]

        colors_other_list = []
        loc_npa = other_npa
        i=0
        for i in range(len(other_npa)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size==0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g=np.median(
                img0[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b=np.median(
                img0[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            colors_other_list.append((r,g,b))
        if len(colors_other_list)>0:
            red = (200, 0, 10)
            t1 = np.array(colors_other_list) - red
            mask = ~np.isnan(t1).any(axis=1)
            loc_npa = loc_npa[mask]
            t1=t1[mask]
            loc_red_other = loc_npa[np.sum(np.power(t1,2),axis=1)<3000]

    return len(loc_red), len(loc_white), len(loc_black), len(loc_red_bw), len(loc_red_other)

def get_num_of_enemy_internal_parallel(img0):
    threshold = 0.10
    template1 = cv2.imread('signs/sign_I_2.png', 0)
    template2 = cv2.imread('signs/sign_I_3.png', 0)

    w, h = template1.shape[::-1]
    img_grey = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    _ = cv2.rectangle(img_grey, (0, 720), (368, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (690, 0), (1228, 36), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (0, 0), (68, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1854, 0), (1920, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1800, 1000), (1920, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (750, 1048), (1385, 1090), (0, 0, 0), -1)

    templates = (template1,template2)
    with ThreadPool(2) as pool:
        res_matchs = pool.starmap(cv2.matchTemplate, zip(repeat(img_grey),templates,repeat(cv2.TM_SQDIFF_NORMED)))

    #res_match1 = cv2.matchTemplate(img_grey, template1, cv2.TM_SQDIFF_NORMED)
    #res_match2 = cv2.matchTemplate(img_grey, template2, cv2.TM_SQDIFF_NORMED)
    locs = (np.where(res_match <= threshold) for res_match in res_matchs)
    loc1, loc2 = (list(zip(*x[::-1])) for x in locs)

    loc=loc1+loc2
    loc = list(set(loc))


    if len(loc) > 0:
        loc = remove_close_jit(loc)
    else:
        loc = np.array([])

    loc_npa=np.array(loc)

    loc_red = []
    loc_white = []
    loc_black = []
    loc_red_bw = []
    loc_red_other = []

    if len(loc) > 0:
        colors_list = []

        i=0
        for i in range(len(loc)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]

            if len(t1)==0:
                colors_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g=np.median(
                    img0[:,:,1][(loc_npa[i,1]+73):(loc_npa[i,1]+91),loc_npa[i,0]:(loc_npa[i,0]+w)]
            )
            b=np.median(
                    img0[:,:,0][(loc_npa[i,1]+73):(loc_npa[i,1]+91),loc_npa[i,0]:(loc_npa[i,0]+w)]
            )

            colors_list.append((r,g,b))
        good_colors = np.array(((200, 0, 10),(229,229,226)))

        red = (200, 0, 10)
        t1=np.array(colors_list) - red
        mask1 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_red = loc_npa[mask1]

        white = (229,229,226)
        t1=np.array(colors_list) - white
        mask2 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_white = loc_npa[mask2]

        black = (4,2,3)
        t1=np.array(colors_list) - black
        mask3 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_black = loc_npa[mask3]

        other_npa = loc_npa[~mask1&~mask2&~mask3].copy()

        b_w = np.concatenate((loc_white,loc_black))
        colors_bw_list = []

        loc_npa = b_w
        i=2
        for i in range(len(b_w)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size==0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g=np.median(
                img0[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b=np.median(
                img0[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            colors_bw_list.append((r,g,b))
        if len(colors_bw_list)>0:
            red = (200, 0, 10)
            t1 = np.array(colors_bw_list) - red
            mask = ~np.isnan(t1).any(axis=1)
            loc_npa = loc_npa[mask]
            t1=t1[mask]
            loc_red_bw = loc_npa[np.sum(np.power(t1,2),axis=1)<3000]

        colors_other_list = []
        loc_npa = other_npa
        i=0
        for i in range(len(other_npa)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size==0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g=np.median(
                img0[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b=np.median(
                img0[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            colors_other_list.append((r,g,b))
        if len(colors_other_list)>0:
            red = (200, 0, 10)
            t1 = np.array(colors_other_list) - red
            mask = ~np.isnan(t1).any(axis=1)
            loc_npa = loc_npa[mask]
            t1=t1[mask]
            loc_red_other = loc_npa[np.sum(np.power(t1,2),axis=1)<3000]

    return len(loc_red), len(loc_white), len(loc_black), len(loc_red_bw), len(loc_red_other)


def get_num_of_enemy_service(img0):
    threshold = 0.10
    template1 = cv2.imread('signs/sign_I_2.png', 0)
    template2 = cv2.imread('signs/sign_I_3.png', 0)

    w, h = template1.shape[::-1]
    img_grey = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    _ = cv2.rectangle(img_grey, (0, 720), (368, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (690, 0), (1228, 36), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (0, 0), (68, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1854, 0), (1920, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1800, 1000), (1920, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (750, 1048), (1385, 1090), (0, 0, 0), -1)

    templates = (template1, template2)
    with ThreadPool(2) as pool:
        res_matchs = pool.starmap(cv2.matchTemplate, zip(repeat(img_grey), templates, repeat(cv2.TM_SQDIFF_NORMED)))

    # res_match1 = cv2.matchTemplate(img_grey, template1, cv2.TM_SQDIFF_NORMED)
    # res_match2 = cv2.matchTemplate(img_grey, template2, cv2.TM_SQDIFF_NORMED)
    locs = (np.where(res_match <= threshold) for res_match in res_matchs)
    loc1, loc2 = (list(zip(*x[::-1])) for x in locs)

    loc = loc1 + loc2
    loc = list(set(loc))

    if len(loc) > 0:
        loc = remove_close_jit(loc)
    else:
        loc = np.array([])

    loc_npa = np.array(loc)

    loc_red = []
    loc_white = []
    loc_black = []
    loc_red_bw = []
    loc_red_other = []

    if len(loc) > 0:
        colors_list = []

        i = 0
        for i in range(len(loc)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]

            if len(t1) == 0:
                colors_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img0[:, :, 1][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
            )
            b = np.median(
                img0[:, :, 0][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
            )

            colors_list.append((r, g, b))
        good_colors = np.array(((200, 0, 10), (229, 229, 226)))

        red = (200, 0, 10)
        t1 = np.array(colors_list) - red
        mask1 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_red = loc_npa[mask1]

        white = (229, 229, 226)
        t1 = np.array(colors_list) - white
        mask2 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_white = loc_npa[mask2]

        black = (4, 2, 3)
        t1 = np.array(colors_list) - black
        mask3 = np.sum(np.power(t1, 2), axis=1) < 3000
        loc_black = loc_npa[mask3]

        other_npa = loc_npa[~mask1 & ~mask2 & ~mask3].copy()

        b_w = np.concatenate((loc_white, loc_black))
        colors_bw_list = []

        loc_npa = b_w
        i = 2
        for i in range(len(b_w)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img0[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img0[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            colors_bw_list.append((r, g, b))
        if len(colors_bw_list) > 0:
            red = (200, 0, 10)
            t1 = np.array(colors_bw_list) - red
            mask = ~np.isnan(t1).any(axis=1)
            loc_npa = loc_npa[mask]
            t1 = t1[mask]
            loc_red_bw = loc_npa[np.sum(np.power(t1, 2), axis=1) < 3000]

        colors_other_list = []
        loc_npa = other_npa
        i = 0
        for i in range(len(other_npa)):
            t1 = img0[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img0[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img0[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            colors_other_list.append((r, g, b))
        if len(colors_other_list) > 0:
            red = (200, 0, 10)
            t1 = np.array(colors_other_list) - red
            mask = ~np.isnan(t1).any(axis=1)
            loc_npa = loc_npa[mask]
            t1 = t1[mask]
            loc_red_other = loc_npa[np.sum(np.power(t1, 2), axis=1) < 3000]

    return len(loc_red), len(loc_white), len(loc_black), len(loc_red_bw), len(loc_red_other)


# for pt in loc:
#     _ = cv2.rectangle(img_grey, tuple(pt), (pt[0] + w, pt[1] + h), 0, 2)
# plt.imshow(img_grey,cmap = 'gray')

"""
pic_false
    8101 Lamp is over the tank
    8401 Timeline. Tank too big. Only sequence can show that there is a tank
    8551 Timeline. Tank too big. The sign is out of the screen
"""

"""
1071 - 1030
431 - 385

y=(loc_npa[i, 1] + 46,loc_npa[i, 1] + 46 + 14)
x=(loc_npa[i, 0] - 41,loc_npa[i, 0] - 41 + w)
p1=(x[0],y[0])
p2=(x[1],y[1])

_ = cv2.rectangle(img_grey, p1, p2, 0, 2)
plt.imshow(img_grey,cmap = 'gray')
"""