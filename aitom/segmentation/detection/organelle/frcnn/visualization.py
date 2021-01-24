# -*- coding: utf-8 -*-
"""
Created on Sun May 20 22:15:20 2018

@author: Berothy
"""
from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
import pickle
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses_fn
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
import os
from keras_frcnn import resnet as nn
from keras_frcnn.simple_parser import get_data
from keras.models import load_model
from keras.utils import plot_model

with open('config.pickle', 'rb') as f_in:
    cfg = pickle.load(f_in)
cfg.use_horizontal_flips = False
cfg.use_vertical_flips = False
cfg.rot_90 = False
#    cfg.im_size = 1500
#    cfg.img_channel_mean = [150.5608, 150.5608, 150.5608]
model_path = cfg.model_path
class_mapping = cfg.class_mapping
if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
input_shape_img = (None, None, 3)
input_shape_features = (None, None, 1024)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(cfg.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping),
                               trainable=True)
model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)


print('Loading weights from {}'.format(model_path))
model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)


#==============================================================================
# f = open('./modelsummary.txt','w+')
# f.write(model_rpn.summary())
# f.close()
#==============================================================================
plot_model(model_rpn, to_file='model_rpn.png')
plot_model(model_classifier, to_file='model_classifier.png')
#model_classifier.summary()