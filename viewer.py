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

count = 0

class Frame:
    def __init__(self):
        self.timestamp = 0
        self.count = 0
        self.jpg = 0
        self.camfps = 0
        self.streamfps = 0

import time

while running:
    try:
        frame = footage_socket.recv_pyobj()
        source = cv2.imdecode(frame.jpg, 1)

        timestamp_string = datetime.datetime.fromtimestamp(frame.timestamp.timestamp(),datetime.timezone.utc).isoformat()
        cv2.putText(source,timestamp_string,(0,20),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

        cv2.putText(source,"CamFPS : {:.1f}".format(frame.camfps) + " StrFPS : {:.1f}".format(frame.streamfps) + " Frame: " + str(frame.count),(0,40),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
        #cv2.putText(source,"ProcFPS: {:.1f}".format(frame.streamfps) ,(0,60),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

        cv2.imshow("Stream", source)
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
