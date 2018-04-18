import os
import numpy as np
from lxml.etree import Element, SubElement, tostring
import random
import cv2
import shutil
import tqdm

data_root = '/Users/hzzone/Downloads/egohands_data/_LABELLED_SAMPLES'
with open("egohands_data.txt") as f:
    data = f.readlines()


random.shuffle(data)
random.shuffle(data)

test_data = random.sample(data, int(len(data)*0.2))
train_data = list(set(data) - set(test_data))

def trans(data, set_name):

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    os.mkdir(os.path.join(curr_dir, set_name))
    Annotations_dir = os.path.join(curr_dir, set_name, 'Annotations')
    JPEGImages_dir = os.path.join(curr_dir, set_name, 'JPEGImages')
    os.mkdir(Annotations_dir)
    os.mkdir(JPEGImages_dir)

    for each_pic_data in tqdm.tqdm(data):
    # for each_pic_data in data:
        data_list = each_pic_data.strip().split()
        video_id = data_list[0]
        frame_num = str(data_list[1]).zfill(4)
        new_img_name = '{}_{}'.format(video_id, frame_num)
        frame_num = 'frame_{}.jpg'.format(frame_num)


        im_path = os.path.join(data_root, video_id, frame_num)

        boxes = np.reshape(np.array(map(int, data_list[2:])), (-1, 4))
        node_root = Element('annotation')

        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'egohands'

        node_filename = SubElement(node_root, 'filename')
        node_filename.text = new_img_name
        #
        node_size = SubElement(node_root, 'size')
        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'
        node_width = SubElement(node_size, 'width')
        im_height, im_width, channel = cv2.imread(im_path).shape
        node_width.text = str(im_width)
        #
        node_height = SubElement(node_size, 'height')
        node_height.text = str(im_height)
        #
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channel)
        #
        # im = cv2.imread(im_path)
        # for index in range(boxes.shape[0]):
        #     minx, miny, w, h = boxes[index]
        #     cv2.namedWindow("", 0)
        #     cv2.resizeWindow('', 300, 300)
        #     cv2.rectangle(im, (minx, miny), (minx+w-1, miny+h-1), (0, 255, 0), thickness=2)
        #     print(w, h)
        # cv2.imshow('', im)
        # cv2.waitKey(0)

        effective_hands = 0
        for index in range(boxes.shape[0]):
            minx, miny, w, h = boxes[index]
            if min(w, h)<40:
                continue
            effective_hands = effective_hands + 1
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = 'hand'
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(minx)
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(miny)
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(minx+w-1)
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(miny+h-1)

        xml = tostring(node_root, pretty_print=True)
        # if effective_hands == 0:
        #     print(im_path)
        if effective_hands != 0:
            print(im_path)
            with open(Annotations_dir + "/" + new_img_name+'.xml', 'w') as f:
                f.write(xml)
            shutil.copy(im_path, JPEGImages_dir + '/' + new_img_name + '.jpg')

trans(train_data, 'trainval')
trans(test_data, 'test')
