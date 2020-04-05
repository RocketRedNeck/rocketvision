# rocketvision
Demo and packaging of video processing, display, and streaming pipelines

# License
RocketRedNeck and GPLv3
Uses some code from [Ultranalytics YOLOv3 demo](https://github.com/ultralytics/yolov3)

# Dependencies
1. OpenCV 2.x or 4.x (3.x has a different API and not handled dynamically, yet)
2. ZeroMQ (ZMQ)
3. On **linux** v4l2-ctl should be installed for some direct manipulations not current available in OpenCV
4. torch - for instantiating and running NN models
5. torchvision - for access to pretrained models
6. CUDA - if you want to run torch/torchvision models on GPU
7. Pretrain models
  * The shell script is from the original developers at [Ultranalytics](https://github.com/ultralytics/yolov3:). It only works in linux environments. When I get around to it I will create a pycurl equivalent for portability
  * In the meanwhile, if you want, just navigate to **either** of the following paths and manually download the files and place them in the `ultrayolo/weights` folder

	https://drive.google.com/drive/folders/1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0
   
	https://drive.google.com/drive/folders/1hOM94uvh5sfZ3UEKMDJSuBPvJSlF2ZNL
   
# Running the Demo
1. Have two cameras (built-in is usually 0, and a USB camera is usually 1, sometimes the OS will reverse them, working on it)
2. type 'python demo.py' at a command prompt (default streaming available on localhost port 5555)
   1. Exposure Control : 'z' to reduce 'c' to increase (default is autoexposure)
   3. Recording Control : 'r' toggles
   4. Stream Transmission : 't' toggles
3. type 'python viewer.py' at another command prompt (separate terminal, default stream from port 5555)
   1. Can also be on separate machine as long as IP address of viewer is passed to demo.py 

[_**Markdown Cheatsheet**_](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
