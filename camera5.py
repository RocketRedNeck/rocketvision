
import argparse
import cv2
import datetime
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
        # Example: rtsp://admin:beau1191!@192.168.0.15:554//h264Preview_01_sub
        self.cam = cv2.VideoCapture(f'rtsp://admin:beau1191!@192.168.0.15:554//h264Preview_0{self.src}_sub')
        # campipe = f'rtspsrc location="rtsp://admin:beau1191!@192.168.0.15:554//h264Preview_0{self.src}_sub" ! rtph264depay ! h264parse ! avdec_h264 ! video/x-raw, format=(string)BGRx! videoconvert ! appsink'
        # self.cam = cv2.VideoCapture(campipe,cv2.CAP_GSTREAMER)

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

    # mp.set_start_method('spawn')
    # with mp.Pool(processes = 8) as pool:

    #     aso = []
    #     for n in range(1,9):
    #         a = (args.address,args.port,n)
    #         print(a)
    #         aso.append(pool.apply_async(func=process, args=a))

    #     while (True):
    #         # for p in aso:
    #         #     print(p.ready())

    #         print(datetime.datetime.now().strftime("%X"))
    #         time.sleep(1.0)


    # p = []
    # for n in range(1,9):
    #     p.append(mp.Process(target=process, name=f'cam{n}', args=(args.address,args.port,n)))
    #     p[-1].start()

    # while(True):
    #     print('----------------------------')
    #     for px in p:
    #         print(f'{px.name} is {"alive" if px.is_alive() else "DEAD"}')
    #     time.sleep(1.0)
    
