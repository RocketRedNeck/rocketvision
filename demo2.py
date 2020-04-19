#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo3

A much simplified vision display/processing pipeline with streaming and recording

Copyright (c) 2020 - the.RocketRedNeck@gmail.com

RocketRedNeck and GPLv3 License

RocketRedNeck hereby grants license for others to copy and modify this source code for 
whatever purpose other's deem worthy as long as RocketRedNeck is given credit where 
where credit is due and you leave RocketRedNeck out of it for all other nefarious purposes. 

https://github.com/RocketRedNeck/rocketvision/blob/master/LICENSE

**************************************************************************************************** 
"""

# import the necessary packages

# Create arg parser
import argparse
parser = argparse.ArgumentParser()

# Add OPTIONAL IP Address argument
# Specify with "py rocketvision.py -ip <viewer address> -p <viewer port> "
# viewer address = localhost (default)
# viewer port = 5555 (default)
parser.add_argument('--address', required=False, default='localhost', 
help='IP Address Of Viewer')
parser.add_argument('--port', required=False, default='5555', 
help='Port for Sending to Viewer')

# Parse the args
args = vars(parser.parse_args())

import cv2      # OpenCV 4.x
import time
import datetime
import zmq
import os

from threading import Thread
from threading import Event

import torch
torch.no_grad()
torch.cuda.empty_cache()

# import our classes
import rocketvision as rv
from rocketvision.rate import Rate
    
from nada import Nada
from yolo import Yolo

nada = Nada()
yolo = Yolo(img_size=256) # default is 512 which yeilds about 3.8 fps (i7/940MX), 384 --> 5 fps, 256 --> 7 fps

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect('tcp://'+args['address']+':'+args['port'])


width  = 1280
height = 720
displayWidth = 1280
displayHeight = 720
framerate = 30
flipmethod = 2

def gstreamer_pipeline(
    capture_width=width,
    capture_height=height,
    display_width=displayWidth,
    display_height=displayHeight,
    framerate=framerate,
    flip_method=flipmethod,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


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

# And so it begins
print("Starting ROCKET VISION!")

cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

print("Capture appears online!")

        
runTime = 0
nextTime = time.time() + 1

thismodname = os.path.splitext(os.path.basename(__file__))[0]
videofile = cv2.VideoWriter(thismodname + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), framerate, (displayWidth,displayHeight))

count = 0
record = False
stream = False

running = True

camfps = Rate()

camfps.start()

processor = ImageProcessor(yolo)
processor.start()

procCount = 0
meta = []

while (running):

    if (time.time() > nextTime):
        nextTime = nextTime + 1
        runTime = runTime + 1

    ret_val, img = cam.read()
    timestamp = datetime.datetime.now()     # Approximate time of frame
    count += 1
    camfps.update()

    # TODO: Pass image to processor and get previous meta
    # If the image processor is busy it will simply ignore this image
    # and return the previous meta
    (procCount, meta) = processor.process(img,count)
    processor.overlay(meta, img)

    timestamp_string = datetime.datetime.fromtimestamp(timestamp.timestamp(),datetime.timezone.utc).isoformat()
    cv2.putText(img,timestamp_string,(0,20),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

    cv2.putText(img,"CamFPS : {:.1f}".format(camfps.fps()) + " Frame: " + str(count),(0,40),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
    cv2.putText(img,"ProcFPS: {:.1f}".format(processor.fps.fps()) + " Frame: " + str(procCount) + " (" + str(procCount - count) +")" ,(0,60),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

    if stream == True:
        encoded, buffer = cv2.imencode('.jpg', img)
        socket.send(buffer)

    if record == True:
        videofile.write(img)

    cv2.imshow('main',img)

    key = cv2.waitKey(1)
    if key == 27:
        running = False
    elif key == ord('r'):
        record = not record
        print("RECORDING = ", record)
    elif key == ord('s'):
        stream = not stream
        print("STREAMING = ", stream)

        
# do a bit of cleanup
processor.stop()
cv2.destroyAllWindows()

print("Goodbye!")

