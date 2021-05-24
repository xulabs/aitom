'''
# Reference:
Adapted from code contributed by Yann Henon in 2017 (https://github.com/yhenon/keras-frcnn).
'''
from __future__ import division
import os
import cv2
import numpy as np
import pickle
import time
import math
import datetime
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import argparse
import os
import keras_frcnn.resnet as nn
from keras_frcnn.visualize import draw_boxes_and_label_on_image_cv2
from keras_frcnn.simple_parser import get_data
import matplotlib.pyplot as plt 

def format_img_size(img, cfg):
    """ formats the image size based on config """
    img_min_side = float(cfg.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, cfg):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    img /= cfg.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2,w,h):
    real_x1 = max(int(round(x1 // ratio)),1)
    real_y1 = max(int(round(y1 // ratio)),1)
    real_x2 = min(int(round(x2 // ratio)),w)
    real_y2 = min(int(round(y2 // ratio)),h)

    return real_x1, real_y1, real_x2, real_y2


def predict_single_image(img_path, model_rpn, model_classifier_only, cfg, class_mapping):
    st = datetime.datetime.now()
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    if img is None:
        print('reading image failed.')
        exit(0)

#==============================================================================
#     #add a bilateralfilter & histequalize(this part is now moved to another code)   
#     img = cv2.bilateralFilter(img,9,75,75)
#     img = cv2.equalizeHist(img) 
#     img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#==============================================================================
    X, ratio = format_img(img, cfg)
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
    [Y1, Y2, F] = model_rpn.predict(X)   
    
#==============================================================================
#     # get the feature maps and output from the RPN
#     model_conv1 = Model(inputs=model_rpn.input,  
#                                      outputs=model_rpn.get_layer('activation_40').output)
#     feature1 = model_conv1.predict(X)
#     print(feature1.shape)
#     #show the feature maps
#     images_per_row = int(math.sqrt(feature1.shape[-1]))
#     n_features = feature1.shape[-1]
#     feature_sizex = feature1.shape[1]
#     feature_sizey = feature1.shape[2]
#     n_cols = n_features // images_per_row
#     display_grid = np.zeros((feature_sizex * n_cols, images_per_row * feature_sizey))
#     for col in range(n_cols):
#         for row in range(images_per_row):
#             channel_image = feature1[0,:, :,col * images_per_row + row].copy()
#             # Post-process the feature to make it visually palatable
#             channel_image -= channel_image.mean()
#             channel_image /= channel_image.std()
#             channel_image *= 64
#             channel_image += 128
#             channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * feature_sizex : (col + 1) * feature_sizex,
#                          row * feature_sizey : (row + 1) * feature_sizey] = channel_image
# 
#     # Display the grid
#     scalex = 1. / feature_sizex
#     scaley = 1. / feature_sizey
#     plt.figure(figsize=(scaley * display_grid.shape[1],
#                         scalex * display_grid.shape[0]))
#     print(display_grid.shape)
#     plt.title('feature_map')
#     plt.grid(False)
#     plt.imshow(display_grid, aspect='equal', cmap='viridis')
#     plt.show()
#==============================================================================

    # this is result contains all boxes, which is [x1, y1, x2, y2]
    result = roi_helpers.rpn_to_roi(Y1, Y2, cfg, K.image_dim_ordering(), overlap_thresh=0.7)
    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    result[:, 2] -= result[:, 0]
    result[:, 3] -= result[:, 1]
    bbox_threshold = 0.8

    # apply the spatial pyramid pooling to the proposed regions
    boxes = dict()
    for jk in range(result.shape[0] // cfg.num_rois + 1):
        rois = np.expand_dims(result[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)
        if rois.shape[1] == 0:
            break
        if jk == result.shape[0] // cfg.num_rois:
            # pad R
            curr_shape = rois.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
            rois = rois_padded

        [p_cls, p_regr] = model_classifier_only.predict([F, rois])
#        print(p_cls)
        for ii in range(p_cls.shape[1]):
            if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue

            cls_num = np.argmax(p_cls[0, ii, :])
            if cls_num not in boxes.keys():
                boxes[cls_num] = []
            (x, y, w, h) = rois[0, ii, :]
            try:
                (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except Exception as e:
                print(e)
                pass
            boxes[cls_num].append(
                [cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h),
                 np.max(p_cls[0, ii, :])])
    # add some nms to reduce many boxes
    for cls_num, box in boxes.items():
        boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.4)
        boxes[cls_num] = boxes_nms
        print(class_mapping[cls_num] + ":")
        for b in boxes_nms:
            b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3],width,height)
            print('{} prob: {}'.format(b[0: 4], b[-1]))
    img = draw_boxes_and_label_on_image_cv2(img, class_mapping, boxes)
    print('Elapsed time = {}'.format(datetime.datetime.now() - st))
    result_path = './result_images/{}.jpg'.format('.'.join(os.path.basename(img_path).split('.')[:-1]))
    print('result saved into ', result_path)
#    cv2.imwrite(result_path, img)
    #draw groundtruth box
    all_images = open('mito_simple_label_d+e.txt','r')  
    for image in all_images:
        image = image.strip()
        [filepath,x1,y1,x2,y2,cls_name]=image.split(',')
        if img_path.split('\\')[-1] == filepath.split('\\')[-1]:     
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            print('ground truth bbox: [{},{},{},{}]'.format(x1,y1,x2,y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 8)
            text_label = '{}'.format(cls_name)
            (ret_val, base_line) = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            text_org = (x1, y1 - 0)
            cv2.rectangle(img, (text_org[0] - 1, text_org[1] + base_line - 80),
                          (text_org[0] + ret_val[0] + 120, text_org[1] - ret_val[1] + 40), (0,0,255), 1)
            # this rectangle for fill text rect
            cv2.rectangle(img, (text_org[0] - 1, text_org[1] + base_line - 80),
                          (text_org[0] + ret_val[0] + 120, text_org[1] - ret_val[1] + 40),
                          (0,0,255), -1)
            cv2.putText(img, text_label, text_org, cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
            cv2.imwrite(result_path, img)
    img = cv2.resize(img,(600,621),interpolation=cv2.INTER_CUBIC)
    cv2.imshow('result',img)
    cv2.waitKey(5000)
    all_images.close()
#    return boxes_nms


def predict(args_):
    path = args_.path
    with open('config.pickle', 'rb') as f_in:
        cfg = pickle.load(f_in)
    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False
    if args_.model_path == 'None':
        model_path = cfg.model_path
    else:
        model_path = args_.model_path

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

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    if os.path.isdir(path):
        for idx, img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)
            predict_single_image(os.path.join(path, img_name), model_rpn,
                                 model_classifier_only, cfg, class_mapping)
#==============================================================================
#         #the results of a tomogram can be output to a .txt file
#         tomo_name = path.split('\\')[-1]
#         tomo_coor = path + '\\' + tomo_name + '.txt'
#         coor = []
#         for idx, img_name in enumerate(sorted(os.listdir(path))):
#             if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
#                 continue
#             print(img_name)
#             res = predict_single_image(os.path.join(path, img_name), model_rpn,
#                                  model_classifier_only, cfg, class_mapping)
#             ind = img_name.split('.')[-2].split('_')[-1]
#             for box in res:
#                 coor.append([int(ind),float(box[0]),float(box[1]),float(box[2]),float(box[3]),float(box[-1])])
#         coor = np.array(coor)
#         coor_arg = np.argsort(coor[:,0])
#         coor = coor[coor_arg]     
#         print(coor)
#         print(coor.shape)
#         np.savetxt(tomo_coor,coor,'%d,%d,%d,%d,%d,%.3f',delimiter=" ")
#==============================================================================

    elif os.path.isfile(path):
        print('predict image from {}'.format(path))
        predict_single_image(path, model_rpn, model_classifier_only, cfg, class_mapping)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='./preprocessed_images', help='image path')
    parser.add_argument('--model_path', '-m', default='./model/mito_frcnn_d+e_epoch16_loss_0.31393223867199227.hdf5', help='model path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args)