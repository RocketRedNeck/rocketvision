import cv2

#gst = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

#gst  = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=21/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
gst = "nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=21/1, format=NV12' ! nvvidconv flip-method=2 ! 'video/x-raw,width=1280, height=720' ! nvvidconv ! nvegltransform ! appsink"

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=2),cv2.CAP_GSTREAMER)
key = 0
while key != 27:
    _, frame = cam.read()
    cv2.imshow("camera", frame)
    key = cv2.waitKey(1)
    
