
import copy
import cv2
import zmq
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

# Parse the args
args = vars(parser.parse_args())

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://'+args['address']+':'+args['port'])
footage_socket.setsockopt_string(zmq.SUBSCRIBE, '')

footage_socket.RCVTIMEO = 1000 # in milliseconds
count = 0
running = True

from rocketvision import Rate
fps = Rate()
fps.start()

from nada import Nada
from yolo import Yolo
from resnet50 import ResNet50

nada = Nada()

#nn = ResNet50()
nn = Yolo(img_size=256, conf_thres = 0.5) # default is 512 which yeilds about 3.8 fps (i7/940MX), 384 --> 5 fps, 256 --> 7 fps
# nn = Yolo(cfg='ultrayolo/cfg/yolov3-tiny.cfg', \
#                weights='ultrayolo/weights/yolov3-tiny.pt', \
#                conf_thres = 0.2)



count = 0

class Frame:
    def __init__(self):
        self.timestamp = 0
        self.count = 0
        self.img = 0
        self.camfps = 0
        self.streamfps = 0
        self.srcid = 0

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
        self._srcid = 0
        self.srcid = self._srcid
        self._meta = []
        self.meta = self._meta.copy()
        self._img = []
        self.outimg = self._img.copy()
        self._timestamp = 0
        self.timestamp = self._timestamp
        self.event = Event()
        self.fps = Rate()

        self.history = {}
    
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

    def process(self, source, srcid, count, timestamp):

        if not self.event.isSet():

            #print(f"Triggering on CAM {srcid} - FRAME {count}")
            # copy previous meta data and start a new processing cycle
            self.outimg = self._img.copy()
            self.count = copy.copy(self._count)
            self.meta = self._meta.copy()
            self.srcid = copy.copy(self._srcid)
            self.timestamp = copy.copy(self._timestamp)

            # New cycle
            self._timestamp = timestamp
            self._img = source
            self._count = count
            self._srcid = srcid

            self.event.set()

        if self.srcid in self.history:
            history = self.history[self.srcid]

            if count == history[0] and timestamp == history[1]:
                # already processed it
                self.meta = []
 
        if self.meta != []:
            self.history.update({self.srcid : (self.count, self.timestamp, self.outimg, self.meta)})
        return (self.srcid, self.count, self.timestamp, self.outimg, self.meta)

    def update(self):
        print("ImageProcessor STARTED!")
        self.fps.start()
        self.stopped = False
        self.running = True
        while (self.running):
            if self.event.wait(0.250):
                #print(f"IMAGE PROCESSING frame {self._count}")
                self._meta = self._process.process(source0 = self._img, overlay = False)
                #print(f"Frame {self._count} Processed")
                self.fps.update()
                self.event.clear()

        self.stopped = True
        print("ImageProcessor STOPPED")

    def overlay(self, meta, source):
        self._process.overlay(meta, source)
    def list_overlay(self, meta, srcid, count, timestamp):
        self._process.list_overlay(meta, srcid, count, timestamp)

processor = ImageProcessor(nn)
processor.start()
lastimage = None
lastsrc = 0
while running:
    try:
        frame = footage_socket.recv_pyobj()
        #now_string = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp(),datetime.timezone.utc).isoformat()
        #source = cv2.imdecode(frame.jpg, 1)

        # TODO: Pass image to processor and get previous meta
        # If the image processor is busy it will simply ignore this image
        # and return the previous meta
        (srcid, procCount, timestamp, outimg, meta) = processor.process(frame.img, frame.srcid, frame.count, frame.timestamp)
        if len(meta) > 0:
            processor.overlay(meta, outimg)
            lastimage = outimg.copy()
            lastsrc = copy.copy(srcid)
            processor.list_overlay(meta, srcid, procCount, timestamp)

        # cv2.putText(source,"REC_T: " + now_string,(0,20),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

        # timestamp_string = datetime.datetime.fromtimestamp(frame.timestamp.timestamp(),datetime.timezone.utc).isoformat()
        # cv2.putText(source,"CAM_T: " + timestamp_string,(0,40),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

        # cv2.putText(source,"CAM_F: {:.1f}".format(frame.camfps) + " StrFPS : {:.1f}".format(frame.streamfps) + " Frame: " + str(frame.count),(0,60),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

        #cv2.putText(source,"SRC = " +str(frame.src) + " PRC_F: {:.1f}".format(processor.fps.fps()) + " Frame: " + str(procCount) + " (" + str(procCount - frame.count) +")" ,(0,80),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

        for srcid in processor.history.keys():
            cv2.imshow(f'CAM {srcid}', processor.history[srcid][2])
        # if lastimage is not None:
        #     cv2.imshow(f'CAM {lastsrc}', lastimage)
        key = cv2.waitKey(1)
        fps.update()

        count += 1
        # if (0 == count % 10):
        #     print(f'LOCAL {fps.fps()} : STREAM {frame.streamfps} : DELAY {time.time() - frame.timestamp}')

        if key == 27:
            running = False

    except KeyboardInterrupt:
        break
    except zmq.error.Again:
        count +=1
        print("Waiting... ", count)

cv2.destroyAllWindows()
