from unittest import TestCase

import time

from find_service import find_service
from multiprocessing import Pool, Process
import multiprocessing as mp
import zmq
import cv2
import msgpack
import msgpack_numpy as m
import pandas as pd
from os import listdir
from line_profiler import LineProfiler
from multiprocessing.dummy import Pool as ThreadPool
m.patch()

def send_pic_file(filename,prof=False):
    img = cv2.imread(filename, 1)
    if prof:
        p = Process(target=find_service.main_prof)
    else:
        p = Process(target=find_service.main)
    try:
        p.start()
        # ZeroMQ Context
        context = zmq.Context()
        # Define the socket using the "Context"
        sock = context.socket(zmq.REQ)
        sock.connect("tcp://127.0.0.1:5555")

        message = msgpack.packb(img)
        #message = img.tobytes()
        #print(message)
        # Send a "message" using the socket
        sock.send(message, zmq.NOBLOCK)
        sock.RCVTIMEO = 3000
        msg_res = sock.recv()
        res = msgpack.unpackb(msg_res, use_list=False)
        if prof:
            sock.send("prof".encode(), zmq.NOBLOCK)
            res = sock.recv().decode()
        p.terminate()
        return res
    except:
        p.terminate()
        raise

class TestFind_service(TestCase):
    def test_00check_process(self):
        q = mp.Queue()
        p=Process(target=find_service.test_f,args=("bob",q))
        try:

            p.start()
            self.assertEquals(q.get(), "hello bob")
            p.terminate()
        except:
            p.terminate()
            raise

    def test_01zmq(self):
        p = Process(target=find_service.main)
        try:
            p.start()

            # ZeroMQ Context
            context = zmq.Context()
            # Define the socket using the "Context"
            sock = context.socket(zmq.REQ)
            sock.connect("tcp://127.0.0.1:5555")

            # Send a "message" using the socket
            sock.send("test".encode(),zmq.NOBLOCK)
            sock.RCVTIMEO = 3000
            res = sock.recv().decode()
            self.assertEquals(res,"echo: test")

            p.terminate()
        except:
            p.terminate()
            raise



    def test_02pic1(self):
        filename = '../scenes/102.png'
        #res = send_pic_file(filename)
        res = send_pic_file(filename)
        self.assertEquals(res,(2,0,0,0,0))
    def test_03pic1(self):
        filename = '../scenes/102.png'
        #res = send_pic_file(filename)
        lp = LineProfiler()
        lp_wrapper = lp(send_pic_file)
        res = lp_wrapper(filename,True)
        lp.print_stats()
        print(res)

    def test_04_direct(self):
        df_files = pd.DataFrame({"name":listdir("../scenes")})
        df_files["fullname"]= "scenes/" + df_files["name"]
        l_imgs=[]
        for filename in df_files["fullname"]:
            filename = "../" + filename
            l_imgs.append(cv2.imread(filename, 1))
        pool = ThreadPool(2)
        l_matches = []
        for img0 in l_imgs:
            start_time = time.time()
            l_matches.append(find_service.get_num_of_enemy_parallel(img0,pool))
            print("%ss" % (time.time() - start_time))

    def test_90_61_file(self):
        df_files = pd.DataFrame({"name":listdir("../scenes")})
        df_files["fullname"]= "scenes/" + df_files["name"]
        l_matches=[]
        for filename in df_files["fullname"]:
            print("\n"+filename)
            filename = "../" + filename
            l_matches.append(send_pic_file(filename))
        l_matches = list(zip(*l_matches))

        df_files["red"] = l_matches[0]
        df_files["white"] = l_matches[1]
        df_files["black"] = l_matches[2]
        df_files["red_bw"] = l_matches[3]
        df_files["red_other"] = l_matches[4]

        df_files2 = pd.read_csv("match.csv")
        self.assertEquals(str(df_files.values),str(df_files2.values))


