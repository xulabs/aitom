'''
Author: Guanan Zhao
'''

# Figure 4: CNN-1 intermediate layer results of Respond-CAM. Each sub-figure is 
# presented in the same way as in Figure 3.

import sys
sys.path.append('../')

import os
output_dir = './output/demo_for_figure_4'
try: # If output_dir does not exist, we make it
    os.makedirs(output_dir)
except: # Otherwise, do nothing
    pass

from keras.models import load_model
cnn_1 = load_model('../data/nfree_CNN_1.h5')

import pickle
with open('../data/test_set_nfree.pickle') as f:
    dj_nfree = pickle.load(f)

import cnn_models as C
import respond_cam as R
import figure_util as F

# Below generating the sub-figures in Figure 4.
i = 1200
cnn = cnn_1
input_data = F.plot_data_3d(dj_nfree[i]['v'],
  os.path.join(output_dir, '%d_input.png' % i))

class_index = C.predict(cnn, [dj_nfree[i]])[0][0]
for j, layer_name in enumerate(('input', 'conv1', 'conv2', 'maxpool1', 'conv3', 'conv4', 'maxpool2')):
    respond_cam = R.respond_cam(cnn, dj_nfree[i]['v'], class_index, layer_name)
    figure = F.plot_data_cam_3d(dj_nfree[i]['v'], respond_cam, 
      os.path.join(output_dir, '%d_layer_%d_%s_respond.png' % (i, j, layer_name)))
