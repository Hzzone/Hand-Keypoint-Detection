import os
from ssd_net import *
import sys
sys.path.insert(0, '../caffe/python')
import xml.dom.minidom
import csv
import re
import time

data_dir = '../data'

def read_xmlfile(file_path):
    DomTree = xml.dom.minidom.parse(file_path)
    annotation = DomTree.documentElement
    objectlist = annotation.getElementsByTagName('object')
    label = file_path.split(os.sep)[-1].strip('.xml')
    boxes = []
    for objects in objectlist:
        bndbox = objects.getElementsByTagName('bndbox')[0]
        xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)
        print(xmin, ymin, xmax, ymax)
        boxes.append([label, xmin, ymin, xmax, ymax, 1])
        # print(bndbox)
    return boxes


def output_gt_label(datatset_name):
    anno_path = os.path.join(data_dir, datatset_name, 'test', 'Annotations')
    # img_dir = os.path.join(data_dir, datatset_name, 'test', 'JPEGImages')
    all_boxes = [['id', 'x1', 'y1', 'x2', 'y2', 'score'], ]
    for root, dirs, files in os.walk(anno_path):
        for xml_file in files:
            xml_file_path = os.path.join(root, xml_file)
            all_boxes.extend(read_xmlfile(xml_file_path))
    with open('../data/gth/{}.csv'.format(datatset_name), 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for box in all_boxes:
            csvwriter.writerow(box)

def output(model_def, model_weights, datatset_name):

    img_dir = os.path.join(data_dir, datatset_name, 'test', 'JPEGImages')
    ssd_net = SSD_NET(model_weights, model_def, GPU_MODE=True, threshold=0.2)

    output_boxes = [['id', 'x1', 'y1', 'x2', 'y2', 'score'], ]


    total_time = 0.0

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img_name = img_name.strip('.jpg')

        image = caffe.io.load_image(img_path)

        start = time.time()

        top_label_indices, top_conf, top_xmin, top_ymin, top_xmax, top_ymax = ssd_net.detect(image)

        total_time = total_time + time.time() - start

        print(img_path)

        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label_indice = top_label_indices[i]

            output_boxes.append([img_name, xmin, ymin, xmax, ymax, score])

            assert label_indice == 1.0


    iter_times = re.findall('VGG_HAND_SSD_300x300_(.*?).caffemodel', model_weights.split(os.sep)[-1])[0]
    print(iter_times)
    output_dir = '../output/{}'.format(iter_times)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, '{}.csv'.format(datatset_name))
    with open(output_file, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for box in output_boxes:
            csvwriter.writerow(box)
    return total_time/len(os.listdir(img_dir))



model_def = '../model/deploy.prototxt'
model_weights = '../model/snapshot/VGG_HAND_SSD_300x300_iter_50000.caffemodel'
# model_path = '../model/snapshot'
# total_time = []
# for model_weights in os.listdir(model_path):
#     if model_weights.endswith('.caffemodel'):
#         total_time.append(output(model_def, os.path.join(model_path, model_weights), 'stanfordhands'))
#         total_time.append(output(model_def, os.path.join(model_path, model_weights), 'egohands'))

print(output(model_def, model_weights, 'stanfordhands'))
print(output(model_def, model_weights, 'egohands'))

# print(total_time)
# output_gt_label('egohands')
# output_gt_label('stanfordhands')
# read_xmlfile('/Users/hzzone/Desktop/Hand-Keypoint-Detection/data/stanfordhands/test/Annotations/VOC2007_1.xml')

