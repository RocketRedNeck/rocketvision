
from ultrayolo.models import *
from ultrayolo.utils.datasets import *
from ultrayolo.utils.utils import *

from torchvision import transforms

import cv2
import torch
import time

import numpy as np

class Yolo:
    """
    An neural network pipeline for object detections
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    """
    
    def __init__(self, \
                cfg='ultrayolo/cfg/yolov3.cfg', \
                names='ultrayolo/data/coco.names', \
                weights='ultrayolo/weights/yolov3.pt', \
                img_size=512, \
                conf_thres=0.3, \
                iou_thres = 0.6, \
                half = False, \
                classes = None, \
                agnostic_nms = False):

        # Capture arguments
        self.cfg = cfg
        self.names = names
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.half = half
        self.classes = classes
        self.agnostic_nms = agnostic_nms

        # Initialize
        self.processor='gpu' if torch.cuda.is_available() else 'cpu'
        self.device = torch_utils.select_device(self.processor)

        # Initialize model
        self.model = Darknet(self.cfg, self.img_size)

        # Load weights
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)

        # Second-stage classifier (TBD)
        self.classify = False
        if self.classify:
            self.modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            self.modelc.to(self.device).eval()

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Eval mode
        self.model.to(self.device).eval()

        # Half precision
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = load_classes(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))] 

        self.lastframe = None
        self.firstpass = False      

    def process(self, source0, overlay = True):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Run inference
        t0 = time.time()

        contours = None
        # if self.firstpass:
        #     # Use frame differences to mark moving objects at the end
        #     difference = cv2.absdiff(self.lastframe, source0)
        #     grayscale = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        #     blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
        #     _, threshold = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY)
        #     dilated = cv2.dilate(threshold, None, iterations=10)
        #     contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.lastframe = source0.copy()
        self.firstpass = True

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=True, interp=cv2.INTER_LINEAR)[0] for x in [source0]]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
    
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = self.model(img)[0].float() if self.half else self.model(img)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, self.im0s)

        # Process detections
        self.meta = []
        for i, det in enumerate(pred):  # detections per image
            s = '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    self.meta.append([xyxy, label, cls])
                
                if overlay:
                    self.overlay(self.meta,source0)

            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))

        # # make moving objects "glow"
        # if contours != None:
        #     cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        #     for contour in cntsSorted[:10]:
        #         area = cv2.contourArea(contour)
        #         if area < 950:
        #             continue

        #         polyframe = source0.copy()
        #         cv2.fillPoly(polyframe, pts =contours, color=(0,255,0))
        #         cv2.addWeighted(source0,0.75,polyframe,0.25,0,dst=source0)
        #         #cv2.drawContours(source0, contours, -1, (255, 255, 0), 3)
        #         #(x, y, w, h) = cv2.boundingRect(contour)
        #         #cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return self.meta

    def overlay(self, meta, img):
        for xyxy, label, cls in meta:
            plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
            #plot_one_tag(xyxy, img, label=label, color=self.colors[int(cls)])
