
import argparse
import copy
import cv2
import datetime
import json
import numpy as np
import operator
import os
import psutil
import pynvml # from pip install nvidia-ml-py
import signal
from subprocess import Popen, PIPE
import sys
from threading import Thread, Event
import time
import torch
import zmq

from rocketvision import Rate
from yolo import Yolo
#from resnet50 import ResNet50



# Create arg parser
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

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config_file(file):
    with open(os.path.join(FILE_DIR, file)) as path:
        return json.load(path)

config = load_config_file("hare.json")['hare_options']
camera_list = config['camera_list']
zmq_port = config['zmq_port']

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://'+'*'+':'+zmq_port)
footage_socket.setsockopt_string(zmq.SUBSCRIBE, '')
footage_socket.set_hwm(2)

footage_socket.RCVTIMEO = 1000 # in milliseconds
count = 0
running = True

fps = Rate()
fps.start()

torch.cuda.empty_cache()
pynvml.nvmlInit()
gpuObj = pynvml.nvmlDeviceGetHandleByIndex(0)


#nn = ResNet50()
nn =  [ Yolo(img_size=256, conf_thres = 0.60) # default is 512 which yeilds about 3.8 fps (i7/940MX), 384 --> 5 fps, 256 --> 7 fps
      , Yolo(img_size=256, conf_thres = 0.60)
      , Yolo(img_size=256, conf_thres = 0.60)
      , Yolo(img_size=256, conf_thres = 0.60)
      , Yolo(img_size=256, conf_thres = 0.60)
      , Yolo(img_size=256, conf_thres = 0.60)
      , Yolo(img_size=256, conf_thres = 0.60)
      , Yolo(img_size=256, conf_thres = 0.60)
    ]
# nn = Yolo(cfg='ultrayolo/cfg/yolov3-tiny.cfg', \
#                weights='ultrayolo/weights/yolov3-tiny.pt', \
#                conf_thres = 0.2)

# nn2 = Yolo(cfg='ultrayolo/cfg/yolov3-tiny.cfg', \
#                weights='ultrayolo/weights/yolov3-tiny.pt', \
#                conf_thres = 0.2)


count = 0

# define a function for vertically  
# concatenating images of the  
# same size  and horizontally 
def concat_vh(list_2d): 
    
      # return final image 
    return cv2.vconcat([cv2.hconcat(list_h)  
                        for list_h in list_2d]) 

class Frame:
    def __init__(self):
        self.timestamp = 0
        self.count = 0
        self.img = 0
        self.camfps = 0
        self.streamfps = 0
        self.srcid = 0

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

        next_frame = False
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

            next_frame = True

            self.event.set()

        if self.srcid in self.history:
            history = copy.copy(self.history[self.srcid])

            if count == history[0] and timestamp == history[1]:
                # already processed it
                self.meta = []
 
        if self.meta != []:
            self.history.update({self.srcid : (self.count, self.timestamp, self.outimg, self.meta)})
            #self.list_overlay(self.meta, self.srcid, self.count, self.timestamp)

        return next_frame

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

    def overlay_reticle(self, meta, img, scale, timestamp):
        self._process.overlay_reticle(meta=meta, img=img, scale=scale, timestamp = timestamp)

    def list_overlay(self, meta, srcid, count, timestamp):
        self._process.list_overlay(meta, srcid, count, timestamp)

processor = [ ImageProcessor(nn[0])
            , ImageProcessor(nn[1])
            , ImageProcessor(nn[2])
            , ImageProcessor(nn[3])
            , ImageProcessor(nn[4])
            , ImageProcessor(nn[5])
            , ImageProcessor(nn[6])
            , ImageProcessor(nn[7])
            ]
for p in processor:
    p.start()

images = [[None, None, None],
          [None, None, None],
          [None, None, None]
         ]
scale = 0.66

camera_processes = []
for i in camera_list:
    gstcmd = ['gst-launch-1.0', 
              'rtspsrc',
              f'location="rtsp://SecurityBunny:WhiteRabbit8@192.168.0.15:554//h264Preview_0{i}_sub"', 
              'protocols=GST_RTSP_LOWER_TRANS_TCP',
              'latency=0',
              '!',
              'rtph264depay',
              '!',
              'h264parse',
              '!',
              'decodebin',
              '!',
              'autovideoconvert'
              '!',
              'autovideosink']

    p = Popen(gstcmd,
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                universal_newlines=True,
                bufsize=0
                ) # Windows only: creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    camera_processes.append(p)


latency_dict = [{},{},{},{},{},{},{},{}]
meta = []
fps_time = time.perf_counter() + 1.0
while running:
    try:
        frame = footage_socket.recv_pyobj()
        if images[0][0] is None:
            h,w,d = frame.img.shape
            z = np.zeros((int(h*scale),int(w*scale),d),dtype='uint8')
            images = [[z, z, z],
                      [z, z, z],
                      [z, z, z]
                     ]
        r = (frame.srcid-1) // 3
        c = (frame.srcid-1) % 3

        src_idx = frame.srcid - 1 # & 1

        if w is not None and w > 0:
            images[r][c] = cv2.resize(frame.img, (int(w*scale),int(h*scale)))
            if frame.srcid in processor[src_idx].history:
                metah = processor[src_idx].history[frame.srcid][3]
                timestamp = processor[src_idx].history[frame.srcid][1]
                if len(metah) > 0:
                    if datetime.datetime.now().timestamp() - timestamp.timestamp() < 3.0:
                        processor[src_idx].overlay_reticle(meta = metah, img = images[r][c], scale = scale, timestamp = timestamp)        

            # If the image processor is busy it will simply ignore this image
            # and return the previous meta
            # The oldest stream is processed first, ensuring nothing is stale
            # The average latency will be time to scan all channels
            if frame.srcid not in latency_dict[src_idx]:
                latency_dict[src_idx].update({frame.srcid:0.0})

            # src_list = sorted(latency_dict[src_idx].items(), key=operator.itemgetter(1), reverse=False)

            # if frame.srcid == src_list[0][0]:
            if processor[src_idx].process(frame.img, frame.srcid, frame.count, frame.timestamp):
                latency_string = f'{datetime.datetime.now().timestamp() - latency_dict[src_idx][frame.srcid]:.3f}'
                latency_dict[src_idx].update({frame.srcid:frame.timestamp.timestamp()})
                r = (frame.srcid-1) // 3
                c = (frame.srcid-1) % 3
                cv2.putText(images[r][c],
                            latency_string,
                            (int(225*scale),int(340*scale)),
                            cv2.FONT_HERSHEY_DUPLEX,
                            scale,
                            (0,0,255),
                            int(1*scale))          
                cv2.rectangle(images[r][c],
                              (0,0),
                              (images[r][c].shape[1],images[r][c].shape[0]), 
                              (0,0,255), 
                              2)

        # function calling 
        img_tile = concat_vh(images)
        cv2.imshow('ALL CAMS', img_tile) 

        key = cv2.waitKey(1)
        fps.update()

        count += 1

        if (time.perf_counter() > fps_time):
            fps_time += 1.0
            font_scale = 1.75
            font = cv2.FONT_HERSHEY_SIMPLEX
            images[2][2] = copy.copy(z)
            cv2.putText(images[2][2],
                        f'FPS  = {fps.fps():.1f}',
                        (int(0*scale),int(30*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))

            percs = psutil.cpu_percent(percpu=True)
            cv2.putText(images[2][2],
                        f'CPU % = {percs}',
                        (int(0*scale),int(60*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))

            freqs = psutil.cpu_freq(percpu=True)
            freqs = [f'{f.current/1000:.1f}' for f in freqs]
            cv2.putText(images[2][2],
                        f'CPU f = {freqs}',
                        (int(0*scale),int(90*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))

            loads = psutil.getloadavg()
            cv2.putText(images[2][2],
                        f'LOAD  = {loads}',
                        (int(0*scale),int(120*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))

            temps = psutil.sensors_temperatures()
            package_temp = temps['coretemp'][0].current
            package_limit = temps['coretemp'][0].high
            cv2.putText(images[2][2],
                        f'CPU T = {package_temp:.1f} C',
                        (int(0*scale),int(150*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0) if package_temp < package_limit else (0,255,255),
                        int(1*scale))

            gpu_percs = pynvml.nvmlDeviceGetUtilizationRates(gpuObj)
            gpu_mem   = pynvml.nvmlDeviceGetMemoryInfo(gpuObj)
            cv2.putText(images[2][2],
                        f'GPU % = Time : {gpu_percs.gpu}  Mem : {100*gpu_mem.used/gpu_mem.total:.0f}',
                        (int(0*scale),int(180*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))


            gpu_temp = pynvml.nvmlDeviceGetTemperature(gpuObj, pynvml.NVML_TEMPERATURE_GPU)
            package_limit = 93                        
            cv2.putText(images[2][2],
                        f'GPU T = {gpu_temp:.1f} C',
                        (int(0*scale),int(210*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0) if gpu_temp < package_limit else (0,255,255),
                        int(1*scale))


        if key == 27:
            running = False

    except KeyboardInterrupt:
        break
    except zmq.error.Again:
        count +=1
        print("Waiting... ", count)

cv2.destroyAllWindows()
pynvml.nvmlShutdown()
for p in camera_processes:
    p.send_signal(signal.SIGINT)
