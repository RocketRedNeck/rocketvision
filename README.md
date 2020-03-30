# rocketvision
Demo and packaging of video processing, display, and streaming pipelines

# Dependencies
1. OpenCV 2.x or 4.x (3.x has a different API and not handled dynamically
2. ZeroMQ (ZMQ)
3. On **linux** v4l2-ctl should be installed for some direct manipulations not current available in OpenCV

# Running the Demo
1. Have two cameras (built-in is usually 0, and a USB camera is usually 1)
2. type 'python demo.py' at a command prompt (default streaming available on localhost port 5555)
   1. 'z' to reduce exposure
   2. 'c' to increase exposure
   3. 'r' toggles recording
   4. 't' toggles stream transmission
3. type 'python viewer.py' at another command prompt (separate terminal, default stream from port 5555)
   1. Can also be on separate machine as long as IP address of viewer is passed to demo.py 
