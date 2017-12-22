import os
import os.path as osp
import scipy.io as sio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
import numpy.random as npr
import shutil


test_data = "test_data"
train_data = "training_data"
validation_data = "validation_data"
data_sources = [test_data, train_data, validation_data]
annotations = "annotations"

