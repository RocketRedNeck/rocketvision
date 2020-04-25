import cv2
import zmq
import numpy as np

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

# from turbojpeg import TurboJPEG
# jpeg = TurboJPEG()

from rocketvision import Rate
fps = Rate()
fps.start()

count = 0
while running:
    try:
        frame = footage_socket.recv()
        npimg = np.fromstring(frame, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        # source = jpeg.decode(npimg)
        cv2.imshow("Stream", source)
        key = cv2.waitKey(1)
        fps.update()

        count += 1
        if (0 == count % 10):
            print(fps.fps())

        if key == 27:
            running = False

    except KeyboardInterrupt:
        break
    except zmq.error.Again:
        count +=1
        print("Waiting... ", count)

cv2.destroyAllWindows()
