import os.path as osp
import scipy.io as sio
import os
import numpy as np
import cv2
from lxml.etree import Element, SubElement, tostring
import shutil

test_data = ['/Users/hzzone/Downloads/hand_dataset/test_dataset/test_data']
trainval_data = ['/Users/hzzone/Downloads/hand_dataset/training_dataset/training_data', '/Users/hzzone/Downloads/hand_dataset/validation_dataset/validation_data']
def trans(data_sources, set_name):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    os.mkdir(os.path.join(curr_dir, set_name))
    Annotations_dir = os.path.join(curr_dir, set_name, 'Annotations')
    JPEGImages_dir = os.path.join(curr_dir, set_name, 'JPEGImages')
    os.mkdir(Annotations_dir)
    os.mkdir(JPEGImages_dir)
    # cv2.namedWindow("", 0)
    # cv2.resizeWindow('', 300, 300)
    for each_source in data_sources:
        annotations_source = osp.join(each_source, 'annotations')
        img_source = osp.join(each_source, 'images')
        for mat_file in os.listdir(annotations_source):
            mat_file_path = osp.join(annotations_source, mat_file)
            # print(mat_file_path)
            img_file_path = osp.join(img_source, mat_file.rstrip('.mat'))+'.jpg'
            img = cv2.imread(img_file_path)
            boxes_data = sio.loadmat(mat_file_path)["boxes"].flatten()


            node_root = Element('annotation')

            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'egohands'

            node_filename = SubElement(node_root, 'filename')
            node_filename.text = mat_file.strip('.mat')+'.jpg'
            #
            node_size = SubElement(node_root, 'size')
            node_segmented = SubElement(node_root, 'segmented')
            node_segmented.text = '0'
            node_width = SubElement(node_size, 'width')
            im_height, im_width, channel = img.shape
            node_width.text = str(im_width)
            #
            node_height = SubElement(node_size, 'height')
            node_height.text = str(im_height)
            #
            node_depth = SubElement(node_size, 'depth')
            node_depth.text = str(channel)

            effective_hands = 0
            for box in boxes_data:
                tmp = np.reshape(box[0, 0].tolist()[:4], (-1, 2))
                y1 = int(round(min(tmp[:, 0]), 0))
                y2 = int(round(max(tmp[:, 0]), 0))
                x1 = int(round(min(tmp[:, 1]), 0))
                x2 = int(round(max(tmp[:, 1]), 0))
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                x2 = im_width if x2 > im_width else x2
                y2 = im_height if y2 > im_height else y2
                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1

                width = x2-x1+1
                height = y2-y1+1

                if(min(width, height)<20):
                    continue

                # if x2>im_width or x1<0 or y2>im_height or y1<0:
                #     print(x1, x2, y1, y2, width, height, im_height, im_width)
                    # cv2.imshow("", img)
                    # cv2.waitKey(0)
                if x2<=x1 or y2<=y1:
                    print(x1, y1)


                effective_hands = effective_hands + 1
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = 'hand'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(x1)
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(y1)
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = str(x2)
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(y2)
            xml = tostring(node_root, pretty_print=True)
            if effective_hands != 0:
                with open(Annotations_dir + "/" + mat_file.rstrip('.mat') +'.xml', 'w') as f:
                    f.write(xml)
                shutil.copy(img_file_path, JPEGImages_dir + '/' + mat_file.rstrip('.mat') + '.jpg')


trans(trainval_data, 'trainval')
trans(test_data, 'test')