import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'caffe/python')
import caffe
from utils.ssd_net import *
import time

cap = cv2.VideoCapture(0)
# width = 720
# height = 480
width = 480
height = 360
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

model_def = '/Users/hzzone/Desktop/Hand-Keypoint-Detection/model/deploy.prototxt'
model_weights = '/Users/hzzone/Desktop/Hand-Keypoint-Detection/model/snapshot/VGG_HAND_SSD_300x300_iter__iter_80000.caffemodel'

ssd_net = SSD_NET(model_weights, model_def)

while True:
    # get a frame
    start_time = time.time()
    ret, frame = cap.read()
    # show a frame
    try:
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")

    top_label_indices, top_conf, top_xmin, top_ymin, top_xmax, top_ymax = ssd_net.detect(image_np/255.0)
    print(image_np.shape)

    print(top_conf)
    print(top_label_indices)
    for i in range(len(top_conf)):
        xmin = int(round(top_xmin[i] * width))
        ymin = int(round(top_ymin[i] * height))
        xmax = int(round(top_xmax[i] * width))
        ymax = int(round(top_ymax[i] * height))
        print(xmin, ymin, xmax, ymax)
        # if np.sum(top_xmin[i]<0) > 0 or np.sum(top_xmax[i]<0) > 0 or np.sum(top_ymin[i]<0) > 0 or np.sum(top_ymax[i]<0) > 0:
        #     continue
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # # time.sleep(0.1)
    fps = 1/(time.time() - start_time)
    cv2.putText(frame, 'FPS: %d' % fps, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("capture", frame)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()