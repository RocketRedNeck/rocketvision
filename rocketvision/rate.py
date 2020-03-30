# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 20:46:25 2017

@author: mtkes
"""

import time

class Rate:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
        self._rate = 0.0

    def start(self):
        # start the timer
        self._numFrames = 0
        self._start = time.time()
        return self

    def reset(self):
        self.start()
        
    def stop(self):
        # stop the timer
        self._end = time.time()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (time.time() - self._start)

    def fps(self):
        # compute the (approximate) frames per second
        if (self._numFrames > 10):
            self._rate = self._numFrames / self.elapsed()
            self.reset()
            
        return self._rate


