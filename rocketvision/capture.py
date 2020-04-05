# -*- coding: utf-8 -*-

## NOTE: OpenCV interface to camera controls is sketchy
## use v4l2-ctl directly for explicit control
## example for dark picture: v4l2-ctl -c exposure_auto=1 -c exposure_absolute=10

# import the necessary packages

import cv2

from subprocess import call
from threading import Lock
from threading import Thread
from threading import Condition

import platform
import datetime

import numpy as np

import platform
import subprocess

# import our classes
from rocketvision.rate import Rate
from rocketvision.duration import Duration

class Capture:
    def __init__(self,name,src,width,height,exposure,set_fps=30):

        # Default fps to 30

        print("Creating Capture for " + name)
        
        self._lock = Lock()
        self._condition = Condition()
        self.fps = Rate()
        self.set_fps = set_fps
        self.duration = Duration()
        self.name = name
        self.exposure = exposure
        self.iso = 800
        self.brightness = 1
        self.src = src
        self.width = width
        self.height = height
            
        # initialize the variable used to indicate if the thread should
        # be stopped
        self._stop = False
        self.stopped = True

        self.grabbed = False
        self.frame = None
        self.timestamp = "timestamp_goes_here"
        self.outFrame = None
        self.count = 0
        self.outCount = self.count

        self.monochrome = False

        print("Capture created for " + self.name)

    def start(self):

        
        # start the thread to read frames from the video stream
        print("STARTING Capture for " + self.name)
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        print("Capture for " + self.name + " RUNNING")

        # keep looping infinitely until the thread is stopped
        self.stopped = False
        self.fps.start()

        lastExposure = self.exposure

        if platform.system() == "Linux":
            cmd = ['v4l2-ctl', '--device='+str(self.src),'--list-formats-ext']
            returned_output = subprocess.check_output(cmd)
            print(returned_output.decode("utf-8"))

            cmd = ['v4l2-ctl', '--list-ctrls']
            returned_output = subprocess.check_output(cmd)
            print(returned_output.decode("utf-8"))


        self.camera = cv2.VideoCapture(self.src,apiPreference=cv2.CAP_ANY)

        # OpenCV VideoCapture properties that can be set()
        # CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
        # CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
        # CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
        # CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
        # CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
        # CV_CAP_PROP_FPS Frame rate.
        # CV_CAP_PROP_FOURCC 4-character code of codec.
        # CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
        # CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
        # CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
        # CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
        # CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
        # CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
        # CV_CAP_PROP_HUE Hue of the image (only for cameras).
        # CV_CAP_PROP_GAIN Gain of the image (only for cameras).
        # CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
        # CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
        # CV_CAP_PROP_WHITE_BALANCE_U The U value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
        # CV_CAP_PROP_WHITE_BALANCE_V The V value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
        # CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
        # CV_CAP_PROP_ISO_SPEED The ISO speed of the camera (note: only supported by DC1394 v 2.x backend currently)
        # CV_CAP_PROP_BUFFERSIZE Amount of frames stored in internal buffer memory (note: only supported by DC1394 v 2.x backend currently)

        print("SETTINGS: ",self.camera.get(cv2.CAP_PROP_SETTINGS))
        print("FORMAT: ",self.camera.get(cv2.CAP_PROP_FORMAT))
        print("MODE:", self.camera.get(cv2.CAP_PROP_MODE))
        print("CHANNEL:", self.camera.get(cv2.CAP_PROP_CHANNEL))
        print("AUTOFOCUS:", self.camera.get(cv2.CAP_PROP_AUTOFOCUS))
        print("AUTOEXP:", self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        self.exposure = self.camera.get(cv2.CAP_PROP_EXPOSURE)
        print("EXPOSURE:", self.exposure)
        print("PIXFMT:",self.camera.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))

        if platform.system() == "Linux":
            cmd = ['v4l2-ctl', '-V']
            returned_output = subprocess.check_output(cmd)
            print(returned_output.decode("utf-8"))

        # print("----------------------")
        # self.camera.set(cv2.CAP_PROP_CHANNEL,1)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        # print("CHANNEL:", self.camera.get(cv2.CAP_PROP_CHANNEL))
        # print("AUTOFOCUS:", self.camera.get(cv2.CAP_PROP_AUTOFOCUS))
        # print("AUTOEXP:", self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE))


        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH), self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
        # cmd = ['v4l2-ctl', '--set-fmt-video=pixelformat=MJPG']
        # returned_output = subprocess.check_output(cmd)
        # print(returned_output.decode("utf-8"))
        if platform.system() == "Linux":        
            cmd = ['v4l2-ctl', '-V']
            returned_output = subprocess.check_output(cmd)
            print(returned_output.decode("utf-8"))
            print(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH), self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # self.camera.setPixelFormat(VideoMode.PixelFormat.kYUYV)
        
        self.camera.set(cv2.CAP_PROP_FPS, self.set_fps)

        # self.camera.setBrightness(1)
        #self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)

        # p = self.camera.enumerateVideoModes()
        # for pi in p:
        #     print(pi.fps, pi.height, pi.width, pi.pixelFormat)

        self.setMonochrome(self.monochrome)
            
        count = 0
        while True:
            # if the thread indicator variable is set, stop the thread
            if (self._stop == True):
                self._stop = False
                self.stopped = True
                return
            
            if (lastExposure != self.exposure):
                self.setExposure()
                lastExposure = self.exposure
                
            # Tell the CvSink to grab a frame from the camera and put it
            # in the source image.  If there is an error notify the output.
            #time, img = cvSink.grabFrame(img)
            ret_val, img = self.camera.read()
            timestamp = datetime.datetime.now()    #Close but not exact, need to work out better sync

            if ret_val == 0:
                self._grabbed = False
                # Send the output the error.
                #self.outstream.notifyError(cvSink.getError())
                # skip the rest of the current iteration
                continue

            self._grabbed = True
            self.count = self.count
            
            self.duration.start()
            self.fps.update()
            
            
            # if something was grabbed and retreived then lock
            # the outboundw buffer for the update
            # This limits the blocking to just the copy operations
            # later we may consider a queue or double buffer to
            # minimize blocking
            if (self._grabbed == True):

                timestamp_string = datetime.datetime.fromtimestamp(timestamp.timestamp(),datetime.timezone.utc).isoformat()
                
                self._condition.acquire()
                self._lock.acquire()
                self.count = self.count + 1
                self.grabbed = self._grabbed
                self.frame = img.copy()
                self.timestamp = timestamp_string
                self._lock.release()
                self._condition.notifyAll()
                self._condition.release()

            self.duration.update()

                
        print("Capture for " + self.name + " STOPPING")

    def read(self):
        # return the frame most recently read if the frame
        # is not being updated at this exact moment
        self._condition.acquire()
        self._condition.wait()
        self._condition.release()
        if (self._lock.acquire() == True):
            self.outFrame = self.frame
            self.outCount = self.count
            self.outTimestamp = self.timestamp
            self._lock.release()
            return (self.outFrame, self.outCount, self.outTimestamp, True)
        else:
            return (self.outFrame, self.outCount, "NoTimeStamp", False)

    def processUserCommand(self, key):
        # if key == ord('x'):
        #     return True
        # elif key == ord('d'):
        #     self.contrast+=1
        #     self.stream.set(cv2.CAP_PROP_CONTRAST,self.contrast)
        #     print("CONTRAST = " + str(self.contrast))
        # elif key == ord('a'):
        #     self.contrast-=1
        #     self.stream.set(cv2.CAP_PROP_CONTRAST,self.contrast)
        #     print("CONTRAST = " + str(self.contrast))
        # elif key == ord('e'):
        #     self.saturation+=1
        #     self.stream.set(cv2.CAP_PROP_SATURATION,self.saturation)
        #     print("SATURATION = " + str(self.saturation))
        # elif key == ord('q'):
        #     self.saturation-=1
        #     self.stream.set(cv2.CAP_PROP_SATURATION,self.saturation)
        #     print("SATURATION = " + str(self.saturation))
        # el
        if key == ord('z'):
            self.exposure = self.exposure - 1
            self.setExposure()
            print("EXPOSURE = " + str(self.exposure))
        elif key == ord('c'):
            self.exposure = self.exposure + 1
            self.setExposure()
            print("EXPOSURE = " + str(self.exposure))
        elif key == ord('w'):
            self.brightness+=1
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS,self.brightness)
            print("BRIGHT = " + str(self.brightness))
        elif key == ord('s'):
            self.brightness-=1
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS,self.brightness)
            print("BRIGHT = " + str(self.brightness))
        elif key == ord('p'):
            self.iso = self.iso + 100
            self.camera.set(cv2.CAP_PROP_ISO_SPEED, self.iso)
            print("ISO = " + str(self.iso))
        elif key == ord('i'):
            self.iso = self.iso - 100
            self.camera.set(cv2.CAP_PROP_ISO_SPEED, self.iso)
            print("ISO = " + str(self.iso))
        elif key == ord('m'):
            self.setMonochrome(not self.monochrome)
            print("MONOCHROME = " + str(self.monochrome))

        return False

    def setMonochrome(self, monochrome):
        self.monochrome = monochrome
        self.camera.set(cv2.CAP_PROP_MONOCHROME, 1 if self.monochrome else 0)

    def updateExposure(self, exposure):
        self.exposure = exposure
        
    def setExposure(self):
        self.camera.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        pass
    
    def stop(self):
        # indicate that the thread should be stopped
        self._stop = True
        self._condition.acquire()
        self._condition.notifyAll()
        self._condition.release()

    def isStopped(self):
        return self.stopped
    
