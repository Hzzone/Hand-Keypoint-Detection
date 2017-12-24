import sys
sys.path.append('/Users/HZzone/caffe/python')
import caffe
import numpy as np
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
		self.valid_index = np.where(roi[:,0] != -1)[0]
		self.N = len(self.valid_index)
		self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
		top[0].reshape(1)

	def forward(self,bottom,top):
		self.diff[...] = 0
		top[0].data[...] = 0
		if self.N != 0:
			self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
			top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

	def backward(self,top,propagate_down,bottom):
		for i in range(2):
			if not propagate_down[i] or self.N==0:
				continue
			if i == 0:
				sign = 1
			else:
				sign = -1
			bottom[i].diff[...] = sign * self.diff / bottom[i].num
################################################################################
#############################SendData Layer By Python###########################
################################################################################
class cls_Layer_fc(caffe.Layer):
	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception("Need 2 Inputs")
	def reshape(self, bottom, top):
		label = bottom[1].data
		self.valid_index = np.where(label != -1)[0]
		self.count = len(self.valid_index)
		top[0].reshape(len(bottom[1].data), 2,1,1)
		top[1].reshape(len(bottom[1].data), 1)

	def forward(self, bottom, top):
		top[0].data[...][...] = 0
		top[1].data[...][...] = 0
		top[0].data[0:self.count] = bottom[0].data[self.valid_index]
		top[1].data[0:self.count] = bottom[1].data[self.valid_index]

	def backward(self, top, propagate_down, bottom):
		if propagate_down[0] and self.count !=0:
			bottom[0].diff[...] = 0
			bottom[0].diff[self.valid_index] = top[0].diff[...]
		if propagate_down[1] and self.count != 0:
			bottom[1].diff[...] = 0
			bottom[1].diff[self.valid_index]=top[1].diff[...]
