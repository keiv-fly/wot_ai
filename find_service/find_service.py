import zmq
import sys
import msgpack
import msgpack_numpy as m
from multiprocessing.dummy import Pool as ThreadPool
import cv2
import numba
import numpy as np
from itertools import repeat
import time
from line_profiler import LineProfiler
import io
import find_service.find_close_cy.find_close_cy as fc_cy
import find_service.gray_conv_cy.gray_conv_cy as gc_cy
import find_service.locs_cy.locs_cy as l_cy
import find_service.white_in_part_cy.white_in_part_cy as wip_cy
import hashlib

m.patch()


def test_f(name, q):
    q.put('hello ' + name)


def main():
    # ZeroMQ Context
    context = zmq.Context()

    # Define the socket using the "Context"
    sock = context.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:5555")
    #print("server binded")
    sys.stdout.flush()
    sock.RCVTIMEO = 3000
    pool = ThreadPool(2)
    while True:
        loop_body(sock,pool)


def main_prof():
    # ZeroMQ Context
    context = zmq.Context()

    # Define the socket using the "Context"
    sock = context.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:5555")
    #print("server binded")
    sys.stdout.flush()
    sock.RCVTIMEO = 3000
    pool = ThreadPool(2)
    lp = LineProfiler()
    lp.add_function(get_num_of_enemy_parallel)
    lp_wrapper = lp(loop_body)
    lp_wrapper(sock,pool)
    stream = io.StringIO()
    lp.print_stats(stream=stream)
    message = sock.recv()
    sock.send(stream.getvalue().encode())

def loop_body(sock,pool):
    message = sock.recv()
    # print("server received")
    sys.stdout.flush()
    msg_test = message[:4]
    if msg_test.isalpha() and msg_test.decode() == "test":
        sock.send("echo: test".encode())
    else:

        img = msgpack.unpackb(message)
        # img = np.fromstring(message)
        start_time = time.time()
        # print(img)
        res = get_num_of_enemy_parallel(img, pool)
        print("%ss" % (time.time() - start_time))
        sock.send(msgpack.packb(res))

def loop_body2(sock,pool):
    message = sock.recv()
    # print("server received")
    sys.stdout.flush()
    msg_test = message[:4]
    if msg_test.isalpha() and msg_test.decode() == "test":
        sock.send("echo: test".encode())
    else:

        img = msgpack.unpackb(message)
        # img = np.fromstring(message)
        start_time = time.time()
        # print(img)
        res = get_num_of_enemy_parallel(img, pool)
        print("%ss" % (time.time() - start_time))
        sock.send(msgpack.packb(res))



def get_num_of_enemy_parallel_old(img, pool):
    threshold = 0.10
    template1 = cv2.imread('../signs/sign_I_2.png', 0)
    template2 = cv2.imread('../signs/sign_I_3.png', 0)

    w, h = template1.shape[::-1]
    #img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_grey = (0.299*img[:,:,2]+0.587*img[:,:,1]+0.114*img[:,:,0]).astype(np.uint8)
    img_grey = gc_cy.gray_conv(img)
    _ = cv2.rectangle(img_grey, (0, 720), (368, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (690, 0), (1228, 36), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (0, 0), (68, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1854, 0), (1920, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1800, 1000), (1920, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (750, 1048), (1385, 1090), (0, 0, 0), -1)

    templates = (template1, template2)
    res_matchs = pool.starmap(cv2.matchTemplate, zip(repeat(img_grey), templates, repeat(cv2.TM_SQDIFF_NORMED)))

    #locs = [l_cy.locs_cy(res_match,threshold) for res_match in res_matchs]
    locs = pool.starmap(l_cy.locs_cy,zip(res_matchs, repeat(threshold)))
    #locs = pool.starmap(locs_py, zip(res_matchs, repeat(threshold)))
    #locs = [locs_py(res_match, threshold) for res_match in res_matchs]

    loc = np.concatenate((locs[0], locs[1]))
    loc = [tuple(x) for x in loc]
    loc = list(set(loc))

    if len(loc) > 0:
        loc = np.array(loc).astype(dtype=np.int32)
        #loc = remove_close(loc)
        loc = fc_cy.remove_close(loc)
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
            t1 = img[:, :, 2][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]

            if len(t1) == 0:
                colors_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
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
            t1 = img[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
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
            t1 = img[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
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

def get_num_of_enemy_parallel(img, pool):
    threshold = 0.10
    template1 = cv2.imread('../signs/sign_I_2.png', 0)
    template2 = cv2.imread('../signs/sign_I_3.png', 0)

    w, h = template1.shape[::-1]
    #img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_grey = (0.299*img[:,:,2]+0.587*img[:,:,1]+0.114*img[:,:,0]).astype(np.uint8)
    img_grey = gc_cy.gray_conv(img)
    _ = cv2.rectangle(img_grey, (0, 720), (368, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (690, 0), (1228, 36), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (0, 0), (68, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1854, 0), (1920, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1800, 1000), (1920, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (750, 1048), (1385, 1090), (0, 0, 0), -1)

    #templates = (template1, template2)
    img_grey1 = img_grey[:, :970]
    img_grey2 = img_grey[:,950:]
    #imgs = [img_grey1]*2 + [img_grey2] *2
    pars = list(zip(
        [img_grey1] * 2 + [img_grey2] * 2,
        [template1, template2]*2,
        repeat(cv2.TM_SQDIFF_NORMED)
    ))
    res_matchs = pool.starmap(cv2.matchTemplate, pars)

    locs = pool.starmap(l_cy.locs_cy,zip(res_matchs, repeat(threshold)))
    locs = [np.array(x) for x in locs]
    for i, loc in enumerate(locs):
        if i>=2 and loc.shape[0]>0:
            for j, loc_j in enumerate(loc):
                locs[i][j, 0] = loc_j[0]+950



    loc = np.concatenate(locs)
    loc = [tuple(x) for x in loc]
    loc = list(set(loc))

    if len(loc) > 0:
        loc = np.array(loc).astype(dtype=np.int32)
        #loc = remove_close(loc)
        loc = fc_cy.remove_close(loc)
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
            if loc_npa[i, 1] + 73 > img.shape[0] or (loc_npa[i, 0] + w) > img.shape[1]:
                colors_list.append((1000, 1000, 1000))
                continue
            t1 = img[:, :, 2][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]

            if len(t1) == 0:
                colors_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
            )
            colors_list.append((r, g, b))
        good_colors = np.array(((200, 0, 10), (229, 229, 226)))

        red = (200, 0, 10)
        t1 = np.array(colors_list) - red
        mask1 = np.sum(np.power(t1, 2), axis=1) < 3000
        if np.isnan(t1).any():
            x=1
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
            if loc_npa[i, 1] + 46 > img.shape[0] or (loc_npa[i, 0] - 41 + w) < 0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            t1 = img[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
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
            if loc_npa[i, 1] + 46 > img.shape[0] or (loc_npa[i, 0] - 41 + w) < 0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            t1 = img[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
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

def get_num_of_enemy_parallel2(img, pool):
    threshold = 0.10
    template1 = cv2.imread('../signs/sign_I_2.png', 0)
    template2 = cv2.imread('../signs/sign_I_3.png', 0)

    w, h = template1.shape[::-1]
    #img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_grey = (0.299*img[:,:,2]+0.587*img[:,:,1]+0.114*img[:,:,0]).astype(np.uint8)
    img_grey = gc_cy.gray_conv(img)
    _ = cv2.rectangle(img_grey, (0, 720), (368, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (690, 0), (1228, 36), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (0, 0), (68, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1854, 0), (1920, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1800, 1000), (1920, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (750, 1048), (1385, 1090), (0, 0, 0), -1)


    #l_parts = white_in_part_py(img_grey, 252)
    l_parts = wip_cy.white_in_part_cy(img_grey, 252)
    l_parts = tuple(map(tuple,l_parts))

    img_parts = []
    ni = img_grey.shape[0]//10
    nj = img_grey.shape[1]//10
    for part in l_parts:
        img_parts.append(img_grey[np.maximum(part[0]*ni-14,0):(part[0]+1)*ni+14, np.maximum(part[1]*nj-14,0):(part[1]+1)*nj+14])
    #templates = (template1, template2)
    #img_grey1 = img_grey[:, :970]
    #img_grey2 = img_grey[:,950:]
    #imgs = [img_grey1]*2 + [img_grey2] *2
    pars = list(zip(
        (x for x in img_parts for i in range(2)),
        [template1, template2]*len(img_parts),
        repeat(cv2.TM_SQDIFF_NORMED)
    ))

    res_matchs = pool.starmap(cv2.matchTemplate, pars)

    res_matchs2=[np.array(x).copy() for x in res_matchs]



    locs = pool.starmap(l_cy.locs_cy,zip(res_matchs2, repeat(threshold)))
    locs = [np.array(x) for x in locs]
    #print(hashlib.md5(str(locs).encode()).hexdigest())
    #print(locs)
    for i, loc in enumerate(locs):
        if loc.shape[0]>0:
            for j, loc_j in enumerate(loc):
                locs[i][j, 0] = loc_j[0] + np.maximum(l_parts[i//2][1] * nj-14,0)
                locs[i][j, 1] = loc_j[1] + np.maximum(l_parts[i//2][0] * ni-14,0)




    loc = np.concatenate(locs)
    #print(loc)
    loc = [tuple(x) for x in loc]

    loc = list(set(loc))


    if len(loc) > 0:
        loc = np.array(loc).astype(dtype=np.int32)
        #loc = remove_close(loc)
        loc = fc_cy.remove_close(loc)
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
            if loc_npa[i, 1] + 73 > img.shape[0] or (loc_npa[i, 0] + w) > img.shape[1]:
                colors_list.append((1000, 1000, 1000))
                continue
            t1 = img[:, :, 2][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]

            if len(t1) == 0:
                colors_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
            )
            colors_list.append((r, g, b))
        good_colors = np.array(((200, 0, 10), (229, 229, 226)))

        red = (200, 0, 10)
        t1 = np.array(colors_list) - red
        mask1 = np.sum(np.power(t1, 2), axis=1) < 3000
        if np.isnan(t1).any():
            x=1
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
            if loc_npa[i, 1] + 46 > img.shape[0] or (loc_npa[i, 0] - 41 + w) < 0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            t1 = img[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
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
            if loc_npa[i, 1] + 46 > img.shape[0] or (loc_npa[i, 0] - 41 + w) < 0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            t1 = img[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
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

def get_num_of_enemy_seq(img):
    threshold = 0.10
    template1 = cv2.imread('../signs/sign_I_2.png', 0)
    template2 = cv2.imread('../signs/sign_I_3.png', 0)

    w, h = template1.shape[::-1]
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ = cv2.rectangle(img_grey, (0, 720), (368, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (690, 0), (1228, 36), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (0, 0), (68, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1854, 0), (1920, 274), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (1800, 1000), (1920, 1090), (0, 0, 0), -1)
    _ = cv2.rectangle(img_grey, (750, 1048), (1385, 1090), (0, 0, 0), -1)

    templates = (template1, template2)
    res_match1 = cv2.matchTemplate(img_grey, template1, cv2.TM_SQDIFF_NORMED)
    res_match2 = cv2.matchTemplate(img_grey, template2, cv2.TM_SQDIFF_NORMED)
    res_matchs=(res_match1,res_match2)
    locs = (np.where(res_match <= threshold) for res_match in res_matchs)
    loc1, loc2 = (list(zip(*x[::-1])) for x in locs)

    loc = loc1 + loc2
    loc = list(set(loc))

    if len(loc) > 0:
        loc = remove_close(loc)
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
            t1 = img[:, :, 2][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]

            if len(t1) == 0:
                colors_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 73):(loc_npa[i, 1] + 91), loc_npa[i, 0]:(loc_npa[i, 0] + w)]
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
            t1 = img[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_bw_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
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
            t1 = img[:, :, 2][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                 (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            if t1.size == 0:
                colors_other_list.append((1000, 1000, 1000))
                continue
            r = np.median(t1)
            g = np.median(
                img[:, :, 1][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
                (loc_npa[i, 0] - 41):(loc_npa[i, 0] - 41 + w)]
            )
            b = np.median(
                img[:, :, 0][(loc_npa[i, 1] + 46):(loc_npa[i, 1] + 46 + 14),
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

def remove_close(loc):

    loc_filtered = []

    n = loc.shape[0]
    for i in range(n):
        loc0 = loc[0]
        loc_filtered.append(loc0)
        loc = loc[1:]
        loc = loc[np.abs(np.sum(loc - loc0[np.newaxis, :], axis=1)) > 6]
        if len(loc) == 0:
            break
    return loc_filtered

def locs_py(res_match,threshold):
    loc = np.where(res_match <= threshold)
    loc = list(zip(*loc[::-1]))
    if len(loc)==0:
        loc = np.empty((0,2),dtype=np.int32)
    else:
        loc = np.array(loc)
    return loc

def white_in_part_py(img, threshold = 252):
    ni = img.shape[0]//10
    nj = img.shape[1]//10
    res_ij = False

    l_sq = []
    for i in range(10):
        for j in range(10):
            res_ij = False
            for ii in range(ni):
                if not res_ij:
                    for jj in range(nj):
                        if img[i*ni+ii, j*nj+jj] > threshold:
                            res_ij = True
                            l_sq.append((i, j))
                            break
                else:
                    break
    return l_sq



if __name__ == "__main__":
    main()
