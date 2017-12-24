# -*- coding: utf-8 -*-
import h5py as hy
import random
from tqdm import tqdm
import cv2
import numpy as np
import os.path as osp
import os

def read_txt(data_source, pos_count, part_count, neg_count):
	lines = []
	with open(osp.join(data_source, "pos.txt")) as f:
		lines.extend(np.random.choice(f.readlines(), size=pos_count, replace=False))
	with open(osp.join(data_source, "part.txt")) as f:
		lines.extend(np.random.choice(f.readlines(), size=part_count, replace=False))
	with open(osp.join(data_source, "neg.txt")) as f:
		lines.extend(np.random.choice(f.readlines(), size=neg_count, replace=False))
	return lines


def generate_hdf5(file_list, save_dir, txt_save_path, segment_size=1000):
	'''
	1. 由于hdf5文件大小的限制，每1000张图片生成一个h5文件
	2. 传入的file_list 需要random.choice生成自己需要的大小，不需要全部传入
	3. negative, positive, part需要总的shuffle一次
	:param file_list: 传入的文件读取的file_list, 格式按照roadmap
	:return: None
	'''
	random.shuffle(file_list)
	random.shuffle(file_list)
	file_list = [image_file.split() for image_file in file_list]
	width, height, channels = cv2.imread(file_list[0][0]).shape

	tmp_value = len(file_list)/segment_size*segment_size ## 最后一个1000的位置
	for index in tqdm(range(0, len(file_list), segment_size)):
		if index > tmp_value:
			data = np.zeros(((len(file_list)-index), channels, width, height), dtype=np.float32)
			labels = np.ones((len(file_list)-index), dtype=np.int)
			roi = np.ones((len(file_list)-index, 1, 4), dtype=np.float32)
			pts = np.ones((len(file_list)-index, 1, 42), dtype=np.float32)
			for i in range(index, len(file_list)):
				im = cv2.imread(file_list[index][0])
				im = (im - 127.5) / 127.5
				data[i % segment_size, :, :] = np.transpose(im, (2, 0, 1))
				labels[i % segment_size] = int(file_list[index][1])
				roi[i % segment_size, :, :] = map(float, file_list[index][2: 5])
				pts[i % segment_size, :, :] = map(float, file_list[index][6: 47])
		else:
			data = np.zeros((segment_size, channels, width, height), dtype=np.float32)
			labels = np.ones((len(file_list)-index), dtype=np.int)
			roi = np.ones((segment_size, 1, 4), dtype=np.float32)
			pts = np.ones((segment_size, 1, 42), dtype=np.float32)
			for i in range(index, index+segment_size):
				im = cv2.imread(file_list[index][0])
				im = (im - 127.5) / 127.5
				data[i % segment_size, :, :] = np.transpose(im, (2, 0, 1))
				labels[i % segment_size] = int(file_list[index][1])
				roi[i % segment_size, :, :] = np.reshape(map(float, file_list[index][2: 6]), (-1, 4))
				pts[i % segment_size, :, :] = np.reshape(map(float, file_list[index][6: 49]), (-1, 42))
		with hy.File(osp.join(save_dir, str(index/segment_size)+'.h5'), 'w') as h5_file:
			h5_file['data'] = data
			h5_file['label'] = labels

	temp_str = ""
	for h5_file in os.listdir(save_dir):
		temp_str += osp.join(save_dir, h5_file) + "\n"
	with open(txt_save_path, "w") as f:
		f.write(temp_str)



if __name__ == "__main__":
	lines = read_txt("/Users/HZzone/Desktop/Hand-Keypoint-Detection/data/p-net/training_data", part_count=30000, pos_count=30000, neg_count=90000)
	generate_hdf5(lines, "/Users/HZzone/Desktop/Hand-Keypoint-Detection/data/p-net/train_h5", "/Users/HZzone/Desktop/Hand-Keypoint-Detection/p-net/train.txt")

