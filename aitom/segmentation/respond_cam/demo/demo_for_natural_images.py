'''
Author: Guanan Zhao
'''

# Experiment in Appendix:
# Comparison of Grad-CAM and Respond-CAM on natural images. These images 
# were selected from PASCAL 2007 test set. The VGG16 network with pre-trained 
# parameters on ImageNet was used.

import sys
sys.path.append('../')

import os
output_dir = './output/demo_for_natural_images'
try: # If output_dir does not exist, we make it
    os.makedirs(output_dir)
except: # Otherwise, do nothing
    pass

input_dir = '../data/natural_images'
image_files = os.listdir(input_dir)

 
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import cv2

import respond_cam as R
import figure_util as F

cnn = VGG16(weights='imagenet')

for file in image_files:
    img = image.load_img(os.path.join(input_dir, file), target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    preds = cnn.predict(img_data)
    top_1 = decode_predictions(preds)[0][0]
    print('%s: %s (%s) with probability %.2f' % (file, top_1[1], top_1[0], top_1[2]))

    class_name = top_1[1]
    class_index = np.argmax(preds)
    score_symbol = cnn.layers[-1].output
    grad_cam = R.grad_cam(cnn, img_data, class_index, 'block5_conv3', score_symbol, dim=2)
    respond_cam = R.respond_cam(cnn, img_data, class_index, 'block5_conv3', score_symbol, dim=2)

    raw_img = cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_UNCHANGED)
    F.plot_data_cam_2d(raw_img, grad_cam,
      os.path.join(output_dir, '%s_%s_grad.jpg' % (file, class_name)))
    F.plot_data_cam_2d(raw_img, respond_cam,
      os.path.join(output_dir, '%s_%s_respond.jpg' % (file, class_name)))
    cv2.imwrite(os.path.join(output_dir, '%s_raw.jpg' % file), raw_img)
