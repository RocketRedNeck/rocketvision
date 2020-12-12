
import argparse
import cv2
import datetime
import json
import os
import time
from threading import Thread
from threading import Event
from rocketvision.rate import Rate
import zmq


# Create arg parser
parser = argparse.ArgumentParser()

# Add OPTIONAL IP Address argument
# Specify with "py camera5.py -ip <viewer address> -p <viewer port> "
# viewer address = localhost (default)
# viewer port = 5555 (default)
parser.add_argument('--address', required=False, default='localhost', 
help='IP Address Of Viewer')
parser.add_argument('--port', required=False, default='5555', 
help='Port for Sending to Viewer')
parser.add_argument('--n', required=False, default='1', 
help='Stream Number')


# Parse the args
args = parser.parse_args()
print(args)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config_file(file):
    with open(os.path.join(FILE_DIR, file)) as path:
        return json.load(path)

config = load_config_file("hare.json")['hare_options']
rtsp_user = config['rtsp_user']
password = config['password']
rtsp_ip = config['rtsp_ip']

class Frame:
    def __init__(self):
        self.timestamp = 0
        self.count = 0
        self.img = 0
        self.camfps = 0
        self.streamfps = 0
        self.srcid = 0

class ImageCapture:
    def __init__(self, src = 0):
        self.src = src
        self.cam = cv2.VideoCapture(f'rtsp://{rtsp_user}:{password}@{rtsp_ip}//h264Preview_0{self.src}_sub')

        self.img = None
        self.count = 0
        self.fps = Rate()
        self.timestamp = 0
        self.event = Event()
        self.running = False
    
    def __enter__(self):
        return self
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        print(f'\n\n\n\nReleasing Camera {self.src}')
        self.cam.release()
        
    def start(self, wait = True, timeout = 5.0):        
        # start the thread to read frames from the video stream
        print("STARTING ImageCapture...")
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        start = time.time()
        if wait:
            while not self.isRunning() and ((time.time() - start) <= timeout):
                time.sleep(0.1)

        if not self.isRunning():
            print("WARNING: ImageCapture may not have started!!!")

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
    
    def read(self):
        if self.event.wait(0.250):
            return (self.count, self.img, self.timestamp)
        else:
            return (None, None, None)

    def update(self):
        print("ImageCapture STARTED!")
        self.fps.start()
        self.stopped = False
        self.running = True
        while (self.running):
            ret,self.img = self.cam.read()
            self.timestamp = datetime.datetime.now()
            if (ret):
                self.count += 1
                self.fps.update()
                self.event.set()
                

        self.stopped = True
        print("ImageCapture STOPPED")    


def process(address, port, n):
    print(f'PROCESS = {address}, {port}, {n}')
    with ImageCapture(int(n)) as cam:
        time.sleep(5)
        cam.start()

        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.connect('tcp://'+address+':'+port)
        socket.set_hwm(1)

        stream = True

        running = True

        fps = Rate()
        fps.start()

        lastframecount = 0

        frame = Frame()
        frame.srcid = cam.src

        next_time = time.perf_counter() + 1.0

        while running:
            frame.count, frame.img, frame.timestamp = cam.read()
            frame.camfps = cam.fps.fps()

            if time.perf_counter() > next_time:
                next_time += 1.0
                print(f'FPS = {frame.camfps}')
            
            if (frame.count != None):
                if (frame.count != lastframecount):
                    lastframecount = frame.count
                    if stream:
                        socket.send_pyobj(frame)

                    fps.update()
                    frame.streamfps = fps.fps()

            key = cv2.waitKey(1)
            
            if key == 27:
                running = False
            elif key == ord('s'):
                stream = not stream
    
# import multiprocessing as mp

if __name__ == '__main__':
    process(args.address, args.port, args.n)
    
