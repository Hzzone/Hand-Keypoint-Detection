from __future__ import print_function
import sys
sys.path.insert(0, '../caffe/python')
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess

# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 10 x 10
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    return net






# The database file for training data. Created by data/VOC0712/create_data.sh
train_data = "../data/lmdb/trainval_lmdb"
model_name = "VGG_HAND_SSD_300x300"
# The database file for testing data. Created by data/VOC0712/create_data.sh
# Specify the batch sampler.
resize_width = 300
resize_height = 300
resize = "{}x{}".format(resize_width, resize_height)
batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]
train_transform_param = {
        'mirror': True,
        'mean_value': [127.0, 127.0, 127.0],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'distort_param': {
                'brightness_prob': 0.5,
                'brightness_delta': 32,
                'contrast_prob': 0.5,
                'contrast_lower': 0.5,
                'contrast_upper': 1.5,
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'random_order_prob': 0.0,
                },
        'expand_param': {
                'prob': 0.5,
                'max_expand_ratio': 4.0,
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
use_batchnorm = False
lr_mult = 1

# model definition files.
train_net_file = "train.prototxt"
deploy_net_file = "deploy.prototxt"
solver_file = "solver.prototxt"
# snapshot prefix.
snapshot_prefix = "snapshot/VGG_HAND_SSD_300x300_iter_"
# Stores LabelMapItem.
label_map_file = "../data/labelmap_voc.prototxt"

# MultiBoxLoss parameters.
num_classes = 2
share_location = True
background_label_id=0
train_on_diff_gt = True
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
ignore_cross_boundary_bbox = False
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
    }
loss_param = {
    'normalization': normalization_mode,
    }

# parameters for generating priors.
# minimum dimension of input image
min_dim = 300
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# conv9_2 ==> 1 x 1
mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
# in percent %
min_ratio = 20
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
max_sizes = [min_dim * 20 / 100.] + max_sizes
steps = [8, 16, 32, 64, 100, 300]
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# L2 normalize conv4_3.
normalizations = [20, -1, -1, -1, -1, -1]
# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
flip = True
clip = False


# Divide the mini-batch to different GPUs.
batch_size = 2
num_gpus = 1
iter_size = 20000
device_id = 0
batch_size_per_device = batch_size

batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
solver_mode = P.Solver.GPU

base_lr = 0.0001

solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [80000, 100000, 120000],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 120000,
    'snapshot': 80000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True
    }

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
        train=True, output_label=True, label_map_file=label_map_file,
        transform_param=train_transform_param, batch_sampler=batch_sampler)

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False)

AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult, conf_postfix='hand_detection')

# Create the MultiBoxLossLayer.
name = "mbox_loss"
mbox_layers.append(net.label)
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
        loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)
