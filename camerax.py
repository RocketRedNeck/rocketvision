
import argparse
import cv2
import datetime
import json
import multiprocessing as mp
import os
import sys
import time
from threading import Thread
from threading import Event
import zmq
from rocketvision.rate import Rate
import vid_streamv3 as vs


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

# config = load_config_file("hare.json")['hare_options']
rtsp_user = 'admin' #config['rtsp_user']
password = 'Beau1191!' #config['password']
rtsp_ip = '192.168.0.20' #config['rtsp_ip']
zmq_ip = 0 #config['zmq_ip']
zmq_port = 0 #config['zmq_port']

class Frame:
    def __init__(self):
        self.timestamp = 0
        self.count = 0
        self.img = 0
        self.camfps = 0
        self.streamfps = 0
        self.srcid = 0

'''
Main class
'''
class mainStreamClass:
    def __init__(self, address, port, cam_num, verbose=False):

        self.src = int(cam_num)
        self.count = 0
        self.fps = Rate()
        self.timestamp = 0
        self.address = address
        self.port = port
        self.verbose = verbose

        self.camProcess = None
        self.cam_queue = None
        self.stopbit = None
        self.camlink = f'rtsp://{rtsp_user}:{password}@{rtsp_ip}//h264Preview_0{self.src}_sub'
        self.framerate = 7

    
    def startMain(self):

        #set  queue size
        self.cam_queue = mp.Queue(maxsize=7)

        self.stopbit = mp.Event()
        self.camProcess = vs.StreamCapture(self.camlink,
                             self.stopbit,
                             self.cam_queue,
                             self.framerate)
        self.t = Thread(target=self.camProcess.run)
        self.t.setDaemon=True
        self.t.start()

        # context = zmq.Context()
        # socket = context.socket(zmq.PUB)
        # socket.connect('tcp://'+self.address+':'+self.port)
        # socket.set_hwm(1)

        fps = Rate()
        fps.start()

        lastframecount = 0

        frame = Frame()
        frame.srcid = self.src

        next_time = time.perf_counter() + 1.0

        try:
            self.fps.start()
            while True:

                if not self.cam_queue.empty():
                    # print('Got frame')
                    cmd, val = self.cam_queue.get()
                    self.timestamp = datetime.datetime.now()
                    self.fps.update()

                    # if cmd == vs.StreamCommands.RESOLUTION:
                    #     pass #print(val)

                    if cmd == vs.StreamCommands.FRAME:
                        frame.count += 1
                        frame.img = val
                        frame.timestamp = self.timestamp
                        frame.camfps = self.fps.fps()

                        if time.perf_counter() > next_time:
                            next_time += 1.0
                            if self.verbose:
                                print(f'FPS = {frame.camfps:.2f}  {frame.streamfps:.2f}')

                        if self.verbose and frame.img is not None:
                            cv2.imshow(f'CAM {frame.srcid}', frame.img)
                            cv2.waitKey(1)

                        if frame.count is not None:
                            if frame.count != lastframecount:
                                lastframecount = frame.count
                                #socket.send_pyobj(frame)

                                fps.update()
                                frame.streamfps = fps.fps()
                else:
                    time.sleep(1/self.framerate)

        except KeyboardInterrupt:
            print('Caught Keyboard interrupt')

        except Exception as e:
            print('Caught Main Exception')
            print(e)

        self.stopCamStream()
        cv2.destroyAllWindows()


    def stopCamStream(self):
        print('in stopCamStream')

        if self.stopbit is not None:
            self.stopbit.set()
            while not self.cam_queue.empty():
                try:
                    _ = self.cam_queue.get()
                except:
                    break
                self.cam_queue.close()

            self.camProcess.join()

    def __enter__(self):
        return self
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        print(f'\n\n\n\nReleasing Camera {self.src}')
        try:
            self.stopCamStream()
        except:
            pass

if __name__ == '__main__':
    try:
        mc = mainStreamClass(zmq_ip, zmq_port, args.n, verbose=True)
        mc.startMain()
    except Exception as e:
        print(e)
