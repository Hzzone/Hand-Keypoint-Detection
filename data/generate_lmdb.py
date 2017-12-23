import numpy.random as npr

size = 48
net = str(size)
with open('Proposal-Net/training_data/pos_training_data.txt', 'r') as f:
	pos = f.readlines()

with open('Proposal-Net/training_data/neg_training_data.txt', 'r') as f:
	neg = f.readlines()

with open('Proposal-Net/training_data/part_training_data.txt', 'r') as f:
	part = f.readlines()


def view_bar(num, total):
	rate = float(num) / total
	rate_num = int(rate * 100) + 1
	r = '\r[%s%s]%d%%' % ("#" * rate_num, " " * (100 - rate_num), rate_num,)
	sys.stdout.write(r)
	sys.stdout.flush()


import sys
import cv2
import numpy as np

cls_list = []
cur_ = 0
sum_ = len(pos)
for line in pos:
	view_bar(cur_, sum_)
	cur_ += 1
	words = line.split()
	image_file_name = 'Proposal-Net/training_data/positive/' + words[0] + '.jpg'
	im = cv2.imread(image_file_name)
	h, w, ch = im.shape
	if h != 12 or w != 12:
		im = cv2.resize(im, (12, 12))
	im = np.swapaxes(im, 0, 2)
	im = (im - 127.5) / 127.5
	label = 1
	roi = [-1, -1, -1, -1]
	pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
	cls_list.append([im, label, roi])

cur_ = 0
neg_keep = npr.choice(len(neg), size=120000, replace=False)
sum_ = len(neg_keep)
for i in neg_keep:
	line = neg[i]
	view_bar(cur_, sum_)
	cur_ += 1
	words = line.split()
	image_file_name = 'Proposal-Net/training_data/negative/' + words[0] + '.jpg'
	im = cv2.imread(image_file_name)
	h, w, ch = im.shape
	if h != 12 or w != 12:
		im = cv2.resize(im, (12, 12))
	im = np.swapaxes(im, 0, 2)
	im = (im - 127.5) / 127.5
	label = 0
	roi = [-1, -1, -1, -1]
	pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
	cls_list.append([im, label, roi])
import cPickle as pickle

fid = open("Proposal-Net/training_data/cls.lmdb", 'w')
pickle.dump(cls_list, fid)
fid.close()

roi_list = []
cur_ = 0
part_keep = npr.choice(len(part), size=40000, replace=False)
sum_ = len(part_keep)
for i in part_keep:
	line = part[i]
	view_bar(cur_, sum_)
	cur_ += 1
	words = line.split()
	image_file_name = 'Proposal-Net/training_data/part/' + words[0] + '.jpg'
	im = cv2.imread(image_file_name)
	h, w, ch = im.shape
	if h != 12 or w != 12:
		im = cv2.resize(im, (12, 12))
	im = np.swapaxes(im, 0, 2)
	im -= 128
	label = -1
	roi = [float(words[2]), float(words[3]), float(words[4]), float(words[5])]
	pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
	roi_list.append([im, label, roi])
cur_ = 0
sum_ = len(pos)
for line in pos:
	view_bar(cur_, sum_)
	cur_ += 1
	words = line.split()
	image_file_name = 'Proposal-Net/training_data/positive/' + words[0] + '.jpg'
	im = cv2.imread(image_file_name)
	h, w, ch = im.shape
	if h != 12 or w != 12:
		im = cv2.resize(im, (12, 12))
	im = np.swapaxes(im, 0, 2)
	im = (im - 127.5) / 127.5
	label = -1
	roi = [float(words[2]), float(words[3]), float(words[4]), float(words[5])]
	pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
	roi_list.append([im, label, roi])
import cPickle as pickle

fid = open("Proposal-Net/training_data/roi.lmdb", 'w')
pickle.dump(roi_list, fid)
fid.close()
