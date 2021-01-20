from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys
from . import utils
from . import lfw
import os
import math
import tensorflow.contrib.slim as slim
from . import sphere_network as network
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import importlib
import pdb


def test(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, args.model)
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.test_list_dir))
            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.test_data_dir), pairs,
                                                 args.test_list_dir)
            image_size = args.image_size
            print('image size', image_size)
            images_placeholder = tf.placeholder(tf.float32,
                                                shape=(None, args.image_height, args.image_width,
                                                       args.image_width),
                                                name='image')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            # network definition.
            prelogits1 = network.infer(images_placeholder, args.embedding_size)
            if args.fc_bn:
                print('do batch norm after network')
                prelogits = slim.batch_norm(prelogits1,
                                            is_training=phase_train_placeholder,
                                            epsilon=1e-5,
                                            scale=True,
                                            scope='softmax_bn')
            # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            embeddings = tf.identity(prelogits)
            embedding_size = embeddings.get_shape()[1]
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on testing images')
            batch_size = args.test_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches):
                start_index = i * batch_size
                print('handing {}/{}'.format(start_index, nrof_images))
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = utils.load_data(paths_batch, False, False, args.image_height,args.image_width,False,\
                    (args.image_height,args.image_width))
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                feats, a = sess.run([embeddings, prelogits], feed_dict=feed_dict)
                # do not know for sure whether we should turn this on? it depends.
                feats = utils.l2_normalize(feats)
                emb_array[start_index:end_index, :] = feats

            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                                                                 actual_issame,
                                                                 0.001,
                                                                 nrof_folds=args.test_nrof_folds)
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc) #
            # fill_value="extrapolate"
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)

            tpr1, fpr1, accuracy1, val1, val_std1, far1 = lfw.evaluate(
                emb_array, actual_issame, 0.0001, nrof_folds=args.test_nrof_folds)
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy1), np.std(accuracy1)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val1, val_std1, far1))
            auc = metrics.auc(fpr1, tpr1)
            print('Area Under Curve (AUC): %1.3f' % auc) #
            # fill_value="extrapolate"
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr1, tpr1)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)
