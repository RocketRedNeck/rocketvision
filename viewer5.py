
import argparse
import copy
import cv2
import datetime
from enum import Enum
import json
import numpy as np
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import operator
import os
import psutil
import pynvml # from pip install nvidia-ml-py
import signal
import smtplib, ssl
from subprocess import Popen, PIPE
import sys
from threading import Thread, Event
import time
import torch
import zmq

from rocketvision import Rate
from tee import simple_log
from yolo import Yolo
#from resnet50 import ResNet50

simple_log(__file__)


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
sender_email = config['email_source']
receiver_email = config['email_destination']
password = config['password']

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://'+'*'+':'+zmq_port)
footage_socket.setsockopt_string(zmq.SUBSCRIBE, '')
footage_socket.set_hwm(1)

footage_socket.RCVTIMEO = 1000 # in milliseconds
count = 0
running = True

ssl_port = 465
ssl_context = ssl.create_default_context()


fps = Rate()
fps.start()

torch.cuda.empty_cache()
pynvml.nvmlInit()
gpuObj = pynvml.nvmlDeviceGetHandleByIndex(0)


#nn = ResNet50()

# default is 512 which yeilds about 3.8 fps (i7/940MX), 384 --> 5 fps, 256 --> 7 fps
img_size = 256
conf_thres = 0.2
person_threshold = 0.67
names='ultrayolo/data/coco.names'

cfg='ultrayolo/cfg/yolov3.cfg'
weights='ultrayolo/weights/yolov3.pt'

# cfg='ultrayolo/cfg/yolov3-tiny.cfg'
# weights='ultrayolo/weights/yolov3-tiny.pt'

nn =  [ Yolo(img_size=img_size, conf_thres = conf_thres, cfg = cfg, weights = weights)
      , Yolo(img_size=img_size, conf_thres = conf_thres, cfg = cfg, weights = weights)
      , Yolo(img_size=img_size, conf_thres = conf_thres, cfg = cfg, weights = weights)
      , Yolo(img_size=img_size, conf_thres = conf_thres, cfg = cfg, weights = weights)
      , Yolo(img_size=img_size, conf_thres = conf_thres, cfg = cfg, weights = weights)
      , Yolo(img_size=img_size, conf_thres = conf_thres, cfg = cfg, weights = weights)
      , Yolo(img_size=img_size, conf_thres = conf_thres, cfg = cfg, weights = weights)
      , Yolo(img_size=img_size, conf_thres = conf_thres, cfg = cfg, weights = weights)
    ]

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
                print(f'Already processed {self.srcid} : {count} : {timestamp}')
                self.meta = []
 
        self.history.update({self.srcid : [self.count, self.timestamp, self.outimg, self.meta]})

        return next_frame

    def update(self):
        print("ImageProcessor STARTED!")
        self.fps.start()
        self.stopped = False
        self.running = True
        while (self.running):
            if self.event.wait(0.1):
                self._meta = self._process.process(source0 = self._img, overlay = False)
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
scale = 0.6

camera_file = 'camera7.py'

camera_processes = {}
camera_times = {}
for i in camera_list:
    p = Popen(["python",
               f"{camera_file}", "--n", f"{i}"],
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                universal_newlines=True,
                bufsize=0
                ) # Windows only: creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    camera_processes.update({i:p})
    camera_times.update({i:datetime.datetime.now().timestamp()})


class ReportState(Enum):
    NOTHING = 0
    REPORTED = 1
    LOST = 2

latency_dict = 8*[{}]

report_state = 8*[ReportState.NOTHING]

meta = []
fps_time = time.perf_counter() + 1.0
live_time = fps_time + 3600.00

sms = True
sms_tries = 3

def do_sms(message, image_file = None):
    msg = MIMEMultipart()
    msg['Subject'] = message
    msg['From'] = sender_email
    msg['To'] = ', '.join(receiver_email)

    text = MIMEText(message)
    msg.attach(text)

    if image_file is not None:
        img_data = open(image_file, 'rb').read()
        image = MIMEImage(img_data, name='snapshot')
        msg.attach(image)

    with smtplib.SMTP_SSL("smtp.gmail.com", ssl_port, context=ssl_context) as server:
        for i in range(sms_tries):
            try:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
                #print(f'EMAIL SENT for {message}')
                break
            except Exception as e:
                print(f'[WARNING] {repr(e)}')
                print('Trying SMTP login again')
                time.sleep(1.0)

    if image_file is not None:
        os.remove(image_file)

def thread_sms(message, image = None):
    print(message)
    if sms:
        if image is not None:
            fname = './' + str(time.perf_counter_ns()) + '.png'
            cv2.imwrite(fname,image)
        else:
            fname = None
        t = Thread(target=do_sms, kwargs=dict(message=message, image_file = fname))
        t.daemon = True
        t.start()

thread_sms('SecurityBunny Started')

# Some default to remove some unbounding warnings
h = 480
w = 640
d = 3
z = np.zeros((int(h*scale),int(w*scale),d),dtype='uint8')

cams_ok = False

while running:
    try:
        frame = footage_socket.recv_pyobj()
        now = datetime.datetime.now()
        if images[0][0] is None:
            h,w,d = frame.img.shape
            z = np.zeros((int(h*scale),int(w*scale),d),dtype='uint8')
            images = [[z, z, z],
                      [z, z, z],
                      [z, z, z]
                     ]
        r = (frame.srcid-1) // 3
        c = (frame.srcid-1) % 3

        src_idx = frame.srcid - 1

        camera_times.update({frame.srcid:now.timestamp()})

        if w is not None and w > 0:
            images[r][c] = cv2.resize(frame.img, (int(w*scale),int(h*scale)))
            if frame.srcid in processor[src_idx].history:
                metah = processor[src_idx].history[frame.srcid][3]
                timestamp = processor[src_idx].history[frame.srcid][1]

                if len(metah) > 0:
                    latency = now.timestamp() - timestamp.timestamp()
                    # if the image is more than a few seconds old then displaying
                    # the reticle could be a problem, especially if the detections are moving
                    if latency < 5.0:
                        processor[src_idx].overlay_reticle(meta = metah, img = images[r][c], scale = scale, timestamp = timestamp)
                        people = ['person' in label for x, label, cls in metah]
                        flagged = any(people)
                        if flagged:
                            # There may be people but we should verify the thresholds
                            flagged = False
                            for x, label, cls in metah:
                                if 'person' in label:
                                    try:
                                        x = label.split(' ')
                                        flagged |= float(x[-1]) > person_threshold
                                    except:
                                        pass

                            # If still flagged and appears to be something new we will report it
                            if flagged:
                                if report_state[src_idx] is ReportState.NOTHING:
                                    # This looks new
                                    report_state[src_idx] = ReportState.REPORTED
                                    message = \
f'''Camera {src_idx+1} Alert at {timestamp.strftime("%c")}
{[label for x, label, cls in metah]}
'''
                                    thread_sms(message, image = images[r][c])                                
                                elif report_state[src_idx] is ReportState.LOST:
                                    # We only just lost the track, so just bring
                                    # back the reported stated
                                    print(f'Camera {src_idx + 1} track restored at {now.strftime("%c")}')
                                    report_state[src_idx] = ReportState.REPORTED

                        else:
                            # Nothing was flagged in this frame
                            # It could be a glitch where the subject just briefly disappeared
                            # So demote the track to a lost state
                            # TODO: May add a counter or timer to when lost transitions back
                            # to nothing
                            if report_state[src_idx] is ReportState.REPORTED:
                                print(f'Camera {src_idx + 1} lost track at {now.strftime("%c")}')
                                report_state[src_idx] = ReportState.LOST
                            elif report_state[src_idx] is ReportState.LOST:
                                print(f'Camera {src_idx + 1} cleared at {now.strftime("%c")}')
                                report_state[src_idx] = ReportState.NOTHING
                    else:
                        # Massive latency issue
                        # Track state is invalid
                        print(f'Camera {src_idx + 1} latency is large: {latency}')
                        report_state[src_idx] = ReportState.NOTHING
                else:
                    # No meta data in this frame
                    # It could be a glitch where the subject just briefly disappeared
                    # So demote the track to a lost state
                    # TODO: May add a counter or timer to when lost transitions back
                    # to nothing
                    if report_state[src_idx] is ReportState.REPORTED:
                        print(f'Camera {src_idx + 1} lost track at {now.strftime("%c")}')
                        report_state[src_idx] = ReportState.LOST
                    elif report_state[src_idx] is ReportState.LOST:
                        print(f'Camera {src_idx + 1} cleared at {now.strftime("%c")}')
                        report_state[src_idx] = ReportState.NOTHING

            # If the image processor is busy it will simply ignore this image
            # and return the previous meta
            # The oldest stream is processed first, ensuring nothing is stale
            # The average latency will be time to scan all channels
            if frame.srcid not in latency_dict[src_idx]:
                latency_dict[src_idx].update({frame.srcid:0.0})

            if processor[src_idx].process(frame.img, frame.srcid, frame.count, frame.timestamp):
                color = (0,255,0)
            else:
                color = (0,255,255)

            latency_string = f'{frame.count} : {datetime.datetime.now().timestamp() - latency_dict[src_idx][frame.srcid]:.3f}'
            latency_dict[src_idx].update({frame.srcid:frame.timestamp.timestamp()})
            r = (frame.srcid-1) // 3
            c = (frame.srcid-1) % 3
            cv2.putText(images[r][c],
                        latency_string,
                        (int(225*scale),int(340*scale)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        scale,
                        color,
                        int(1*scale))          
            cv2.rectangle(images[r][c],
                            (0,0),
                            (images[r][c].shape[1],images[r][c].shape[0]), 
                            color, 
                            2)

        # function calling 
        img_tile = concat_vh(images)
        cv2.imshow('ALL CAMS', img_tile) 

        key = cv2.waitKey(1)
        fps.update()

        count += 1

        if (time.perf_counter() > live_time):
            live_time += 3600.0
            thread_sms(f'SecurityBunny NOT DEAD YET')

        if (time.perf_counter() > fps_time):
            for key in camera_times:
                dt = datetime.datetime.now().timestamp() - camera_times[key]
                if dt > 30.0:
                    # This camera has not reported in a while
                    # shut it down and try to restart it
                    camera_processes[key].send_signal(signal.SIGINT)

                    thread_sms(f'SecurityBunny RESTARTING CAMERA {key} : {dt}')

                    p = Popen(["python",
                            f"{camera_file}", "--n", f"{key}"],
                                stdin=PIPE,
                                stdout=PIPE,
                                stderr=PIPE,
                                universal_newlines=True,
                                bufsize=0
                                ) # Windows only: creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                    camera_processes.update({key:p})
                    camera_times.update({key:datetime.datetime.now().timestamp()})

            fps_time += 1.0
            font_scale = 1.75
            font = cv2.FONT_HERSHEY_SIMPLEX
            images[2][2] = copy.copy(z)
            y = 30
            dy = 30
            cv2.putText(images[2][2],
                        f'FPS  = {fps.fps():.1f}',
                        (int(0*scale),int(y*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))

            y += dy
            percs = psutil.cpu_percent(percpu=True)
            cv2.putText(images[2][2],
                        f'CPU % = {percs}',
                        (int(0*scale),int(y*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))

            y += dy
            freqs = psutil.cpu_freq(percpu=True)
            freqs = [f'{f.current/1000:.1f}' for f in freqs]
            cv2.putText(images[2][2],
                        f'CPU f = {freqs}',
                        (int(0*scale),int(y*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))

            y += dy
            loads = psutil.getloadavg()
            cv2.putText(images[2][2],
                        f'LOAD  = {loads}',
                        (int(0*scale),int(y*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))

            y += dy
            mems = psutil.virtual_memory()
            cv2.putText(images[2][2],
                        f'MEM % = {mems[2]}',
                        (int(0*scale),int(y*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))


            y += dy
            temps = psutil.sensors_temperatures()
            package_temp = temps['coretemp'][0].current
            package_limit = temps['coretemp'][0].high
            cv2.putText(images[2][2],
                        f'CPU T = {package_temp:.1f} C',
                        (int(0*scale),int(y*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0) if package_temp < package_limit else (0,255,255),
                        int(1*scale))

            y += dy
            gpu_percs = pynvml.nvmlDeviceGetUtilizationRates(gpuObj)
            gpu_mem   = pynvml.nvmlDeviceGetMemoryInfo(gpuObj)
            cv2.putText(images[2][2],
                        f'GPU % = Time : {gpu_percs.gpu}  Mem : {100*gpu_mem.used/gpu_mem.total:.0f}',
                        (int(0*scale),int(y*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0),
                        int(1*scale))

            y += dy
            gpu_temp = pynvml.nvmlDeviceGetTemperature(gpuObj, pynvml.NVML_TEMPERATURE_GPU)
            package_limit = 93                        
            cv2.putText(images[2][2],
                        f'GPU T = {gpu_temp:.1f} C',
                        (int(0*scale),int(y*scale)),
                        font,
                        scale / font_scale,
                        (0,255,0) if gpu_temp < package_limit else (0,255,255),
                        int(1*scale))


        if key == 27:
            running = False

        cams_ok = True    

    except KeyboardInterrupt:
        break
    except zmq.error.Again:
        if cams_ok:
            count = 0
            cams_ok = False

        count +=1
        print("Waiting... ", count)

        if count > 30:
            count = 0
            if len(camera_processes) > 0:
                for key in camera_processes:
                    camera_processes[key].send_signal(signal.SIGINT)

                time.sleep(1.0)
                thread_sms('SecurityBunny RESTARTING CAMERA PROCESSES')
                camera_processes = {}
                camera_times = {}
                for i in camera_list:
                    p = Popen(["python",
                            f"{camera_file}", "--n", f"{i}"],
                                stdin=PIPE,
                                stdout=PIPE,
                                stderr=PIPE,
                                universal_newlines=True,
                                bufsize=0
                                ) # Windows only: creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                    camera_processes.update({i:p})
                    camera_times.update({i:datetime.datetime.now().timestamp()})


    except Exception as e:
        print(e)
        break

cv2.destroyAllWindows()
pynvml.nvmlShutdown()
for key in camera_processes:
    camera_processes[key].send_signal(signal.SIGINT)
