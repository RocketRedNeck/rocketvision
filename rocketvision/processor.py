
from threading import Lock
from threading import Thread
from threading import Condition

from rocketvision.rate import Rate
from rocketvision.duration import Duration

class Processor:
    def __init__(self,stream,ipdictionary, ipselection):
        print("Creating BucketProcessor for " + stream.name)
        self._lock = Lock()
        self._condition = Condition()
        self.fps = Rate()
        self.duration = Duration()
        self.stream = stream
        self.name = self.stream.name
        self.ipdictionary = ipdictionary
        self.ipselection = ipselection
        self.ip = self.ipdictionary[ipselection]

        self._frame = None
        self.frame = None
        self.count = 0
        self.isNew = False
        
        # initialize the variable used to indicate if the thread should
        # be stopped
        self._stop = False
        self.stopped = True

        print("BucketProcessor created for " + self.name)
        
    def start(self):
        print("STARTING BucketProcessor for " + self.name)
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        print("BucketProcessor for " + self.name + " RUNNING")
        # keep looping infinitely until the thread is stopped
        self.stopped = False
        self.fps.start()

        lastIpSelection = self.ipselection
        
        while True:
            # if the thread indicator variable is set, stop the thread
            if (self._stop == True):
                self._stop = False
                self.stopped = True
                return

            # otherwise, read the next frame from the stream
            # grab the frame from the threaded video stream
            (self._frame, count, timestamp, isNew) = self.stream.read()
            self.duration.start()
            self.fps.update()

            if (lastIpSelection != self.ipselection):
                self.ip = self.ipdictionary[self.ipselection]
                lastIpSelection = self.ipselection

            if (isNew == True):
                # TODO: Insert processing code then forward display changes
                self.ip.process(self._frame)
                
                # Now that image processing is complete, place results
                # into an outgoing buffer to be grabbed at the convenience
                # of the reader
                self._condition.acquire()
                self._lock.acquire()
                self.count = count
                self.isNew = isNew
                self.frame = self._frame
                self.timestamp = timestamp
                self._lock.release()
                self._condition.notifyAll()
                self._condition.release()

            self.duration.update()
                
        print("BucketProcessor for " + self.name + " STOPPING")

    def updateSelection(self, ipselection):
        self.ipselection = ipselection

    def read(self):
        # return the frame most recently processed if the frame
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
            return (self.outFrame, self.outCount, "No Time Stamp", False)
          
    def stop(self):
        # indicate that the thread should be stopped
        self._stop = True
        self._condition.acquire()
        self._condition.notifyAll()
        self._condition.release()

    def isStopped(self):
        return self.stopped
		
