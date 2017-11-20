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


#remove_close_jit = numba.jit(remove_close)

if __name__ == "__main__":
    main()
