import sys
sys.path.append('/Users/HZzone/caffe/python')
import caffe
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S')
################################################################################
#########################ROI Loss Layer By Python###############################
################################################################################
class regression_Layer(caffe.Layer):
	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception("Need 2 Inputs")

	def reshape(self, bottom, top):
		if bottom[0].count != bottom[1].count:
			raise Exception("Input predict and groundTruth should have same dimension")
		roi = bottom[1].data
		self.valid_index = np.where(roi[:, 0] != -1)[0]
		self.N = len(self.valid_index)
		# self.N = np.sum(roi != -1)
		self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
		top[0].reshape(1)

	def forward(self, bottom, top):
		self.diff[...] = 0
		top[0].data[...] = 0
		if self.N != 0:
			self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
			top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

	def backward(self, top, propagate_down, bottom):
		for i in range(2):
			if not propagate_down[i] or self.N == 0:
				continue
			if i == 0:
				sign = 1
			else:
				sign = -1
			bottom[i].diff[...] = sign * self.diff / bottom[i].num

class EuclideanLossLayer(caffe.Layer):
	"""
	Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
	to demonstrate the class interface for developing layers in Python.
	"""

	def setup(self, bottom, top):
		# check input pair
		if len(bottom) != 2:
			raise Exception("Need two inputs to compute distance.")

	def reshape(self, bottom, top):
		# check input dimensions match
		if bottom[0].count != bottom[1].count:
			raise Exception("Inputs must have the same dimension.")
		# difference is shape of inputs
		self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
		# loss output is scalar
		top[0].reshape(1)

	def forward(self, bottom, top):
		self.diff[...] = bottom[0].data - bottom[1].data
		top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.

	def backward(self, top, propagate_down, bottom):
		for i in range(2):
			if not propagate_down[i]:
				continue
			if i == 0:
				sign = 1
			else:
				sign = -1
			bottom[i].diff[...] = sign * self.diff / bottom[i].num

################################################################################
#############################SendData Layer By Python###########################
################################################################################
# class cls_Layer_fc(caffe.Layer):
# 	def setup(self, bottom, top):
# 		if len(bottom) != 2:
# 			raise Exception("Need 2 Inputs")
# 	def reshape(self,bottom,top):
# 		label = bottom[1].data
# 		self.valid_index = np.where(label != -1)[0]
# 		self.count = len(self.valid_index)
# 		top[0].reshape(len(bottom[1].data), 2,1,1)
# 		top[1].reshape(len(bottom[1].data), 1)
# 	def forward(self,bottom,top):
# 		top[0].data[...][...]=0
# 		top[1].data[...][...]=0
# 		top[0].data[0:self.count] = bottom[0].data[self.valid_index]
# 		top[1].data[0:self.count] = bottom[1].data[self.valid_index]
# 	def backward(self,top,propagate_down,bottom):
# 		if propagate_down[0] and self.count!=0:
# 			bottom[0].diff[...]=0
# 			bottom[0].diff[self.valid_index]=top[0].diff[...]
# 			# bottom[0].diff[self.valid_index]=top[0].diff[0:self.count]
# 		if propagate_down[1] and self.count!=0:
# 			bottom[1].diff[...]=0
# 			bottom[1].diff[self.valid_index]=top[1].diff[...]

class filter_Layer(caffe.Layer):
	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception("Need 2 Inputs")
		if len(top) != 2:
			raise Exception("Need 2 Outputs")

	def reshape(self, bottom, top):
		label = bottom[1].data
		self.valid_index = np.where(label != -1)[0]
		self.count = len(self.valid_index)
		# logging.debug("hzzone: %s %s %s %s" % (bottom[0].shape[0], bottom[0].shape[1], bottom[0].shape[2], bottom[0].shape[3]))
		# logging.debug("hzzone: %s %s" % (bottom[1].shape[0], bottom[1].shape[1]))
		# top[0].reshape(self.count, bottom[0].shape[1], bottom[0].shape[2], bottom[0].shape[3])
		if self.count != 0:
			top[0].reshape(self.count, bottom[0].shape[1], bottom[0].shape[2], bottom[0].shape[3])
			top[1].reshape(self.count, bottom[1].shape[1], bottom[1].shape[2], bottom[1].shape[3])
		else:
			top[0].reshape(bottom[0].shape[0], bottom[0].shape[1], bottom[0].shape[2], bottom[0].shape[3])
			top[1].reshape(bottom[1].shape[0], bottom[1].shape[1], bottom[1].shape[2], bottom[1].shape[3])
		# top[1].reshape(self.count, 1)
	def forward(self, bottom, top):
		top[0].data[...] = 0
		# top[0].data[...] = bottom[0].data[self.valid_index]
		top[1].data[...] = 0
		if self.count != 0:
			# top[1].data[...] = 0
			top[0].data[...] = bottom[0].data[self.valid_index]
			top[1].data[...] = bottom[1].data[self.valid_index]
	def backward(self, top, propagate_down, bottom):
		if propagate_down[0] and self.count != 0:
			bottom[0].diff[...] = 0
			bottom[0].diff[self.valid_index] = top[0].diff[...]
		if propagate_down[1] and self.count != 0:
			bottom[1].diff[...] = 0
			bottom[1].diff[self.valid_index] = top[1].diff[...]
