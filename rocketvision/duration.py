# -*- coding: utf-8 -*-

import time

class Duration:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = 0
        self._end = 0
        self._accumulation = 0
        self._numFrames = 0
        self._duration = 0.0

    def start(self):
        # start the timer
        self._start = time.time()
        return self

    def reset(self):
        self._numFrames = 0
        self._accumulation = 0
        
    def stop(self):
        # stop the timer
        self._end = time.time()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self.stop()
        self._accumulation += self.elapsed()
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start)

    def duration(self):
        # compute the average duration
        if (self._numFrames > 10):
            self._duration = self._accumulation / self._numFrames
            self.reset()
            
        return self._duration


