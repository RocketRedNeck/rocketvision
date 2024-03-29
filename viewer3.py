import cv2
#import zmq
import numpy as np
import datetime

import sys

# Create arg parser
import argparse
parser = argparse.ArgumentParser()

# Add OPTIONAL IP Address argument
# Specify with "py bucketvision3.py -ip <viewer address> -p <viewer port> "
# viewer address = localhost (default)
# viewer port = 5555 (default)
parser.add_argument('--address', required=False, default='*', 
help='Interface Address')
parser.add_argument('--port', required=False, default='5555', 
help='Port for Receiving')
parser.add_argument('--n', required=False, default='1')

# Parse the args
args = parser.parse_args()
n = int(args.n)

# context = zmq.Context()
# footage_socket = context.socket(zmq.SUB)
# footage_socket.bind('tcp://'+args['address']+':'+args['port'])
# footage_socket.setsockopt_string(zmq.SUBSCRIBE, '')

# footage_socket.RCVTIMEO = 1000 # in milliseconds
count = 0
running = True

from rocketvision import Rate
fps = Rate()
fps.start()

from nada import Nada
from yolo import Yolo

nada = Nada()
# yolo = Yolo(img_size=256) # default is 512 which yeilds about 3.8 fps (i7/940MX), 384 --> 5 fps, 256 --> 7 fps
yolo = Yolo(cfg='ultrayolo/cfg/yolov3-tiny.cfg', \
               weights='ultrayolo/weights/yolov3-tiny.pt', \
               conf_thres = 0.2)

count = 0

class Frame:
    def __init__(self):
        self.timestamp = 0
        self.count = 0
        self.jpg = 0
        self.camfps = 0
        self.streamfps = 0

from threading import Thread
from threading import Event
import time

class ImageProcessor:
    def __init__(self, process):
        self._process = process
        self.running = False
        self.stopped = True
        self._count = 0
        self.count = self._count
        self._meta = []
        self.meta = self._meta.copy()
        self.event = Event()
        self.fps = Rate()
    
    def start(self, wait = True, timeout = 5.0):        
        # start the thread to read frames from the video stream
        print("STARTING ImageProcess...")
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        start = time.time()
        if wait:
            while not self.isRunning() and ((time.time() - start) <= timeout):
                time.sleep(0.1)

        if not self.isRunning():
            print("WARNING: ImageProcess may not have started!!!")

        return self

    def stop(self, wait = True, timeout = 5.0):
        self.running = False
        start = time.time()
        while not self.stopped and ((time.time() - start) <= timeout):
            time.sleep(0.1)

        if self.isRunning():
            print("WARNING: ImageProcess may not have stopped!!!")
        

    def isRunning(self):
        return self.running

    def process(self, source, count):
        if not self.event.isSet():
            #print(f"Triggering on {count}")
            # copy previous meta data and start a new processing cycle
            self.count = self._count
            self.meta = self._meta.copy()
            self.img = source
            self._count = count
            self.event.set()

        return (self.count, self.meta)

    def update(self):
        print("ImageProcessor STARTED!")
        self.fps.start()
        self.stopped = False
        self.running = True
        while (self.running):
            if self.event.wait(0.250):
                #print(f"IMAGE PROCESSING frame {self._count}")
                self._meta = self._process.process(source0 = self.img, overlay = False)
                #print(f"Frame {self._count} Processed")
                self.fps.update()
                self.event.clear()

        self.stopped = True
        print("ImageProcessor STOPPED")

    def overlay(self, meta, source):
        self._process.overlay(meta, source)

processor = ImageProcessor(yolo)
processor.start()

# cv2.VideoCapture("rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_01_sub")
# cv2.VideoCapture("rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_02_sub")
# cv2.VideoCapture("rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_03_sub")
# cv2.VideoCapture("rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_04_sub")
# cv2.VideoCapture("rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_05_sub")
# cv2.VideoCapture("rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_06_sub")
# cv2.VideoCapture("rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_07_sub")
# cv2.VideoCapture("rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_08_sub")

cap = cv2.VideoCapture(f'rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_0{n}_sub')

framecount = 0
while running:
    try:
        #frame = footage_socket.recv_pyobj()
        ret, frame = cap.read()
        framecount += 1
        now_string = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp(),datetime.timezone.utc).isoformat()
        source = frame

        # TODO: Pass image to processor and get previous meta
        # If the image processor is busy it will simply ignore this image
        # and return the previous meta
        (procCount, meta) = processor.process(source,framecount)
        processor.overlay(meta, source)

        cv2.putText(source,"PRC_F: {:.1f}".format(processor.fps.fps()) + " Frame: " + str(procCount) + " (" + str(procCount - framecount) +")" ,(0,20),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

        cv2.imshow("Stream", source)
        key = cv2.waitKey(1)
        fps.update()

        count += 1

        if key == 27:
            running = False

    except KeyboardInterrupt:
        break
    except Exception as e:
        count += 1
        print(f'{repr(e)} : Waiting... {count}')

cv2.destroyAllWindows()
