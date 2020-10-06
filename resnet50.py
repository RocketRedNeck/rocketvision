
import torchvision
import torch
import cv2
import time

# TODO: Pull in from file
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class ResNet50:
    """
    An neural network pipeline for object detections
    """
    
    def __init__(self, \
                names='ultrayolo/data/coco.names', \
                threshold=0.5, \
                rect_th=3, \
                text_size=3, \
                text_th=3):

        # Capture arguments
        self.names = names
        self.threshold = threshold
        self.rect_th = rect_th
        self.text_size = text_size
        self.text_th = text_th

        # Initialize
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) # Defing PyTorch Transform

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()


    # def get_prediction(self,img_path, threshold):
    #     img = Image.open(img_path) # Load the image
    #     transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
    #     img = transform(img) # Apply the transform to the image
    #     pred = model([img]) # Pass the image to the model
    #     pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    #     pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    #     pred_score = list(pred[0]['scores'].detach().numpy())
    #     pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    #     pred_boxes = pred_boxes[:pred_t+1]
    #     pred_class = pred_class[:pred_t+1]
    #     return pred_boxes, pred_class  

    # def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):

    #     boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
    #     img = cv2.imread(img_path) # Read image with cv2
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    #     for i in range(len(boxes)):
    #         cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    #         cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class

    #     return img


    def process(self, source0, overlay = True):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Run inference
        t0 = time.time()

        img0 = self.transform(source0).cuda()
        pred = self.model([img0])

        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())

        if pred_score != []:
            selected = [pred_score.index(x) for x in pred_score if x > self.threshold]
            if selected != []:
                pred_t = selected[-1] # Get list of index with score greater than threshold.
                pred_boxes = pred_boxes[:pred_t+1]
                pred_class = pred_class[:pred_t+1]

                for i in range(len(pred_boxes)):
                    cv2.rectangle(source0, pred_boxes[i][0], pred_boxes[i][1],color=(0, 255, 0), thickness=self.rect_th) # Draw Rectangle with the coordinates
                    cv2.putText(source0,pred_class[i], pred_boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (0,255,0),thickness=self.text_th) # Write the prediction class

                # if overlay:
                #     self.overlay(self.meta,source0)


