import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'caffe/python')
import caffe
from utils.ssd_net import *
import time
import urllib


## Use local camera
# cap = cv2.VideoCapture(0)
# # width = 720
# # height = 480
width = 640
height = 480
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

## Use ipcam
# url = r"http://192.168.1.190:8080/videofeed"
# capture = cv2.VideoCapture(url)

# Replace the URL with your own IPwebcam shot.jpg IP:port
url = 'http://192.168.1.190:8080/shot.jpg'


model_def = 'model/deploy.prototxt'
model_weights = 'model/snapshot/VGG_HAND_SSD_300x300_iter_50000.caffemodel'

ssd_net = SSD_NET(model_weights, model_def, GPU_MODE=True, threshold=0.7)

while True:
    # get a frame
    # start_time = time.time()
    # ret, frame = capture.read()

    # Use urllib to get the image from the IP camera
    imgResp = urllib.urlopen(url)
    
    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    
    # Finally decode the array to OpenCV usable format ;) 
    frame = cv2.imdecode(imgNp,-1)

    start_time = time.time()

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
        print(xmin, ymin, xmax, ymax, top_conf[i])
        # if np.sum(top_xmin[i]<0) > 0 or np.sum(top_xmax[i]<0) > 0 or np.sum(top_ymin[i]<0) > 0 or np.sum(top_ymax[i]<0) > 0:
        #     continue
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # # time.sleep(0.1)
    fps = 1/(time.time() - start_time)
    cv2.putText(frame, 'FPS: %d' % fps, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("capture", frame)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

# capture.release()
cv2.destroyAllWindows()