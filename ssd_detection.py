import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'caffe/python')
import caffe
from utils.ssd_net import *

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

model_def = '/Users/hzzone/Desktop/Hand-Keypoint-Detection/model/deploy.prototxt'
model_weights = '/Users/hzzone/Desktop/Hand-Keypoint-Detection/model/snapshot/VGG_HAND_SSD_300x300_iter__iter_80000.caffemodel'

ssd_net = SSD_NET(model_weights, model_def)

# image = caffe.io.load_image('/Users/hzzone/Desktop/CARDS_COURTYARD_B_T_0324.jpg')
image = caffe.io.load_image('/Users/hzzone/Desktop/3.jpg')

top_label_indices, top_conf, top_xmin, top_ymin, top_xmax, top_ymax = ssd_net.detect(image)

# print(top_label_indices, top_conf, top_xmin, top_ymin, top_xmax, top_ymax)

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

plt.imshow(image)
currentAxis = plt.gca()

for i in xrange(top_conf.shape[0]):
    xmin = int(round(top_xmin[i] * image.shape[1]))
    ymin = int(round(top_ymin[i] * image.shape[0]))
    xmax = int(round(top_xmax[i] * image.shape[1]))
    ymax = int(round(top_ymax[i] * image.shape[0]))
    score = top_conf[i]
    label = int(top_label_indices[i])
    # label_name = top_labels[i]
    label_name = label
    display_txt = '%s: %.2f' % (label_name, score)
    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    color = colors[label]
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

plt.show()

