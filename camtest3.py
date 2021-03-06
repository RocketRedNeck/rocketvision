
import time

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
parser.add_argument('--n', required=False, default='1', 
help='Stream Number')


# Parse the args
args = parser.parse_args()
print(args)

from threading import Thread
from threading import Event
import time
import datetime
from rocketvision.rate import Rate

import cv2

class ImageCapture:
    def __init__(self, src = 0):
        self.src = src
        self.cam = cv2.VideoCapture(f'rtsp://admin:beaubeau@192.168.0.15:554//h264Preview_0{self.src}_sub')
        self.img = None
        self.count = 0
        self.fps = Rate()
        self.timestamp = 0
        self.event = Event()
        self.running = False
        
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


cam = ImageCapture(int(args.n))
time.sleep(5)
cam.start()

import zmq
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect('tcp://'+args.address+':'+args.port)

stream = True

running = True

fps = Rate()
fps.start()

lastframecount = 0

class Frame:
    def __init__(self):
        self.timestamp = 0
        self.count = 0
        self.img = 0
        self.camfps = 0
        self.streamfps = 0
        self.srcid = 0

frame = Frame()
frame.srcid = cam.src

while running:
    frame.count, frame.img, frame.timestamp = cam.read()
    frame.camfps = cam.fps.fps()
    
    if (frame.count != None):
        if (frame.count != lastframecount):
            lastframecount = frame.count
            if stream:
                #_, frame.jpg = cv2.imencode('.jpeg', img)
                socket.send_pyobj(frame)

            try:    
                cv2.imshow(f'Camera {frame.srcid}', frame.img)
            except Exception as e:
                print(f'[WARNING] : {repr(e)}')            
            fps.update()
            frame.streamfps = fps.fps()
                
    key = cv2.waitKey(1)
    
    if key == 27:
        running = False
    elif key == ord('s'):
        stream = not stream
    
    
