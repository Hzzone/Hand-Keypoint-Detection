import os.path as osp
import os
import shutil
import numpy as np
import cv2
import numpy.random as npr
import sys
sys.path.insert(0, osp.join("..", "utils"))
from utils import IoU
from tqdm import tqdm


p_net = "p-net"
pos = "positive"
part = "part"
neg = 'negative'

def generate_data(data_source, data_save_path, net_size=12):
	if osp.exists(data_save_path):
		shutil.rmtree(data_save_path)
	print data_save_path
	os.mkdir(data_save_path)
	f1 = open(os.path.join(data_save_path, 'pos.txt'), 'w')
	f2 = open(os.path.join(data_save_path, 'neg.txt'), 'w')
	f3 = open(os.path.join(data_save_path, 'part.txt'), 'w')
	pos_save_dir = osp.join(data_save_path, pos)
	part_save_dir = osp.join(data_save_path, part)
	neg_save_dir = osp.join(data_save_path, neg)
	if osp.exists(pos_save_dir):
		shutil.rmtree(pos_save_dir)
	if osp.exists(part_save_dir):
		shutil.rmtree(part_save_dir)
	if osp.exists(neg_save_dir):
		shutil.rmtree(neg_save_dir)
	os.mkdir(pos_save_dir)
	os.mkdir(part_save_dir)
	os.mkdir(neg_save_dir)
	with open(osp.join(data_source, "data.txt"), 'r') as f:
		annotations = f.readlines()
	num = len(annotations)
	print "%d pics in total" % num
	p_idx = 0  # positive
	n_idx = 0  # negative
	d_idx = 0  # dont care
	box_idx = 0
	for annotation in tqdm(annotations):
		annotation = annotation.strip().split(' ')
		file_name = annotation[0]
		bbox = map(float, annotation[1:])
		boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
		img = cv2.imread(osp.join(osp.join(data_source, "images"), file_name + '.jpg'))

		height, width, channel = img.shape

		neg_num = 0
		while neg_num < 30:
			try:
				size = npr.randint(40, min(width, height) / 2)
			except:
				break
			nx = npr.randint(0, width - size)
			ny = npr.randint(0, height - size)
			crop_box = np.array([nx, ny, nx + size, ny + size])

			Iou = IoU(crop_box, boxes)

			cropped_im = img[ny: ny + size, nx: nx + size, :]
			resized_im = cv2.resize(cropped_im, (net_size, net_size), interpolation=cv2.INTER_LINEAR)

			if np.max(Iou) < 0.3:
				# Iou with all gts must below 0.3
				save_file = osp.join(neg_save_dir, "%s.jpg" % n_idx)
				f2.write("%s 0 %s\n" % (save_file, ('-1 '*46).rstrip()))
				cv2.imwrite(save_file, resized_im)
				n_idx += 1
				neg_num += 1

		for box in boxes:
			# box (x_left, y_top, x_right, y_bottom)
			x1, y1, x2, y2 = box
			w = x2 - x1 + 1
			h = y2 - y1 + 1

			# ignore small faces
			# in case the ground truth boxes of small faces are not accurate
			if max(w, h) < 40 or x1 < 0 or y1 < 0:
				continue

			# generate positive examples and part faces
			part_num = 0
			pos_num = 0
			i = 0
			while part_num < 10 or pos_num < 10:
				i += 1
				if i > 49:
					break
				size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

				# delta here is the offset of box center
				delta_x = npr.randint(-w * 0.2, w * 0.2)
				delta_y = npr.randint(-h * 0.2, h * 0.2)

				nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
				ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
				nx2 = nx1 + size
				ny2 = ny1 + size
				# print nx1, ny1, nx2, ny2

				if nx2 > width or ny2 > height:
					continue
				crop_box = np.array([nx1, ny1, nx2, ny2])

				offset_x1 = (x1 - nx1) / float(size)
				offset_y1 = (y1 - ny1) / float(size)
				offset_x2 = (x2 - nx2) / float(size)
				offset_y2 = (y2 - ny2) / float(size)

				cropped_im = img[ny1: ny2, nx1: nx2, :]
				resized_im = cv2.resize(cropped_im, (net_size, net_size), interpolation=cv2.INTER_LINEAR)

				box_ = box.reshape(1, -1)
				if IoU(crop_box, box_) >= 0.65 and pos_num < 10:
					save_file = osp.join(pos_save_dir, "%s.jpg" % p_idx)
					f1.write("%s 1 %.2f %.2f %.2f %.2f %s\n" % (save_file, offset_x1, offset_y1, offset_x2, offset_y2, ('-1 '*42).rstrip()))
					cv2.imwrite(save_file, resized_im)
					p_idx += 1
					pos_num += 1
				elif IoU(crop_box, box_) >= 0.4 and part_num < 10:
					save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
					f3.write("%s -1 %.2f %.2f %.2f %.2f %s\n" % (save_file, offset_x1, offset_y1, offset_x2, offset_y2, ('-1 '*42).rstrip()))
					cv2.imwrite(save_file, resized_im)
					d_idx += 1
					part_num += 1
			box_idx += 1


	f1.close()
	f2.close()
	f3.close()


if __name__ == "__main__":
	generate_data("./detection-dataset/training_data", "/Users/HZzone/Desktop/Hand-Keypoint-Detection/data/p-net/training_data")

