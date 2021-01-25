'''
Author: Guanan Zhao
'''

# Figure 3: Three cases in which Respond-CAM displays significant improvement. In each
# sub-figure, the same 3D image is presented in two ways. 1) Upper half: the parallel
# projection of subtomogram isosurface or Respond-CAM/Grad-CAM contours; 2) lower
# half: the corresponding 2D slices. Notice that Respond-CAM/Grad-CAM heatmaps are
# resized so that they can be overlaid on subtomograms.

import sys
sys.path.append('../')

import os
output_dir = './output/demo_for_figure_3'
try: # If output_dir does not exist, we make it
    os.makedirs(output_dir)
except: # Otherwise, do nothing
    pass

from keras.models import load_model
cnn_1 = load_model('../data/noisy_CNN_1.h5')
cnn_2 = load_model('../data/noisy_CNN_2.h5')

import pickle
with open('../data/test_set_nfree.pickle') as f:
    dj_nfree = pickle.load(f)
with open('../data/test_set_noisy.pickle') as f:
    dj_noisy = pickle.load(f)

import cnn_models as C
import respond_cam as R
import figure_util as F

# Below generating the sub-figures in Figure 3.
for i in (300, 2500, 3300):
    obj_of_interest = F.plot_data_3d(dj_nfree[i]['v'],
      os.path.join(output_dir, '%d_obj.png' % i))
    input_data = F.plot_data_3d(dj_noisy[i]['v'],
      os.path.join(output_dir, '%d_input.png' % i))

    for j, cnn in enumerate((cnn_1, cnn_2)):
        class_index = C.predict(cnn, [dj_noisy[i]])[0][0]

        grad_cam = R.grad_cam(cnn, dj_noisy[i]['v'], class_index, 'maxpool2')
        figure = F.plot_data_cam_3d(dj_noisy[i]['v'], grad_cam, 
          os.path.join(output_dir, '%d_cnn_%d_grad.png' % (i, j+1)))

        respond_cam = R.respond_cam(cnn, dj_noisy[i]['v'], class_index, 'maxpool2')
        figure = F.plot_data_cam_3d(dj_noisy[i]['v'], respond_cam, 
          os.path.join(output_dir, '%d_cnn_%d_respond.png' % (i, j+1)))
        
