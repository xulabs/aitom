# -*- coding=utf-8 -*-
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from tensorflow.compat.v1 import keras
from model import CycleGAN
import keras
from datetime import datetime
import os
import json
import logging
from utils import ImagePool
#from matplotlib import pyplot as plt
import numpy as np
import pickle
#import cPickle as pickle
from dataLoader import *
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from sklearn.metrics import mean_squared_error
from prdc import compute_prdc

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 40, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_integer('max_imgnum', 1600, 'max image number of trainset ')
tf.flags.DEFINE_bool('use_spec_norm', False, 'use spec norm, default: False')
tf.flags.DEFINE_bool('use_wloss', True, 'use wasserstein loss, default: True')

tf.flags.DEFINE_string('X', '../data/same_density.pickle',
                       'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')  # ribosome , membrane, TRiC, proteasome_s
tf.flags.DEFINE_string('Y', '../data/new_4_classes.pickle',
                       'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('load_model', "wgan4_sim_lr2e-5",
                       'folder of saved model that you wish to continue training (e.g. 20190201-1553), default: None')  # ribosome , membrane, TRiC, proteasome_s
# Evaluation Config
tf.flags.DEFINE_string('classification_model', '../data/trainonsub.h5',
                       'the path that the checkpoints of model used to classify, it should be keras model, default: None')
tf.flags.DEFINE_integer('k_prdc', 5, 'the number of nearest neighbour when computing precision, recall, density, and coverage')
# ribosome , membrane, TRiC, proteasome_s
tf.flags.DEFINE_string('new_model', 'gan4_no_wasserstein_lr1e-3', 'new model name')
tf.flags.DEFINE_float('noise', 0, 'noise sigma')
tf.flags.DEFINE_integer('epoch', 10, 'epoch number, default: 256')

def evaluate():
  if FLAGS.load_model.strip() is not None:
    checkpoints_dir = "../checkpoints/" + FLAGS.load_model.strip()
  else:
    print("No checkpoint")
    return

  graph = tf.Graph()
  with graph.as_default():
    cycle_gan = CycleGAN(
        X_train_file=FLAGS.X,
        Y_train_file=FLAGS.Y,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambda1=FLAGS.lambda1,
        lambda2=FLAGS.lambda2,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf,
        use_spec_norm=FLAGS.use_spec_norm,
        use_wloss=FLAGS.use_wloss
    )
    #batch_num = tf.placeholder(tf.int32,shape=[])
    
    G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, cycle_loss, G_gan_loss, F_gan_loss, clip_disc_weights = cycle_gan.model()
    #optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    #summary_op = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver(max_to_keep=1)

  with tf.Session(graph=graph) as sess:
    # sess.run(tf.global_variables_initializer())
    if FLAGS.load_model.strip() is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      best_acc = 5512
      # meta_graph_path = os.path.join(checkpoints_dir, '{}_best_model_{}.ckpt.meta'.format(FLAGS.load_model.strip(),best_acc))
      #epoch = int(meta_graph_path.split("_")[-1].split(".")[0])
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      # restore.restore(sess, './checkpoints/wgan4_sim_best_model_7543.ckpt')
      # restore.restore(sess, os.path.join(checkpoints_dir, '{}_best_model_{}.ckpt'.format(FLAGS.load_model.strip(),best_acc)))
      #step = int(meta_graph_path.split("-")[2].split(".")[0])
      #epoch = 5
    else:
      sess.run(tf.global_variables_initializer())
    
    
    if (FLAGS.image_size==40):
      id2label = {"ribosome": 0, "membrane": 1, "TRiC": 2, "proteasome_s": 3}
    else:
      id2label = {"31": 0, "33": 1, "35": 2, "43": 3, "69": 4, "72": 5,"73":6}
    image_size = FLAGS.image_size
     # ribosome , membrane, TRiC, proteasome_s
    img_den, img_sub, den_label, max_imgnum = load_picdata(
        FLAGS.image_size, FLAGS.X, FLAGS.Y, FLAGS.noise, label=len(id2label))
    # img_den = np.load("/mnt/ssd1/home/v_xindi_wu/proj/dataset/bigdensitymap/400/datasets.npy")
    # img_den = preprocessz(img_den)


    x = img_den.reshape(-1, image_size, image_size, image_size, 1)
    y = img_sub.reshape(-1, image_size, image_size, image_size, 1)
    x_label = den_label

    #print(x[0,:])
    fake_subtomogram = {}
    fake_density = {}
    label2id = {}
    for idx , tag in id2label.items():
      fake_density[idx] = []
      fake_subtomogram[idx] = []
      label2id[tag] = idx

    fake_sub_slice = []

    for step in range(1,x.shape[0]//FLAGS.batch_size+1):
      # get previously generated images
      # fake_y_val = sess.run(fake_y, feed_dict={cycle_gan.x: x[step, :].reshape(-1, image_size, image_size, image_size, 1)})
      fake_y_val, fake_x_val = sess.run([fake_y, fake_x], feed_dict={cycle_gan.x: x[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size, :],
                                                                    cycle_gan.y: y[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size, :]})

      print("step: {}".format(step))
      #print(fake_y_val)
      fake_sub_slice.append(fake_y_val.reshape((-1,image_size,image_size,image_size,1)))
      # fake_subtomogram[label2id[int(x_label[step])]].append(fake_y_val.reshape((image_size,image_size,image_size,1)))
      #fake_density[label2id[int(sub_label[step])]].append(fake_x_val.reshape((image_size,image_size,image_size,1)))
      #print(tag_label[int(label[step])])

    # with open("../result/"+FLAGS.load_model.strip()+"_generate_sub_400.pickle", "wb") as f:
    #   pickle.dump(fake_sub_slice, f)  # py3 load : encoding='iso-8859-1'
    img_fake_sub = np.concatenate(fake_sub_slice, axis=0)
    img_fake_sub = img_fake_sub.reshape((-1,image_size,image_size,image_size,1))
    np.save("../result/{}/best_acc_fake_subtomogram_test".format(checkpoints_dir.split("/")[-1]),img_fake_sub)#1KP8
    np.save("../result/{}/density_map_test".format(checkpoints_dir.split("/")[-1]),x)#1KP8
    np.save("../result/{}/subtomogram_test".format(checkpoints_dir.split("/")[-1]),y)#subtomogram
    np.save("../result/{}/density_map_label_test".format(checkpoints_dir.split("/")[-1]),x_label)
    np.save("../result/{}/subtomogram_label_test".format(checkpoints_dir.split("/")[-1]),x_label)
    
    mse = mean_squared_error(y.reshape((img_fake_sub.shape[0],-1)), img_fake_sub.reshape((img_fake_sub.shape[0],-1)))
    psnr = 10. * np.log10(1. / mse)

    metrics = compute_prdc(real_features=img_sub.reshape((x.shape[0],-1)),
                       fake_features=img_fake_sub.reshape((x.shape[0],-1)),
                       nearest_k=FLAGS.k_prdc)

    # model = load_model(FLAGS.classification_model)
    # print("cls Model Loaded")

    # labels_test = np_utils.to_categorical(den_label, 4)
    # scores = model.evaluate(img_fake_sub, labels_test)
    # print("accuracy: ", scores[1])

    # with open("../result/{}/best_acc_model_test_metrics.json".format(checkpoints_dir.split("/")[-1]),"w", encoding='utf-8') as f:
    #   json.dump({"accuracy":scores[1], "psnr":psnr, **metrics}, f, indent=4)

    with open("../result/{}/best_acc_model_test_metrics.json".format(checkpoints_dir.split("/")[-1]),"w", encoding='utf-8') as f:
      json.dump({"psnr":psnr, **metrics}, f, indent=4)

    return

def uncertainty_est():
  if FLAGS.load_model.strip() is not None:
    checkpoints_dir = "../checkpoints/" + FLAGS.load_model.strip()
  else:
    print("No checkpoint")
    return

  graph = tf.Graph()
  with graph.as_default():
    cycle_gan = CycleGAN(
        X_train_file=FLAGS.X,
        Y_train_file=FLAGS.Y,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambda1=FLAGS.lambda1,
        lambda2=FLAGS.lambda2,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf,
        use_spec_norm=FLAGS.use_spec_norm,
        use_wloss=FLAGS.use_wloss
    )
    #batch_num = tf.placeholder(tf.int32,shape=[])
    
    G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, cycle_loss, G_gan_loss, F_gan_loss, clip_disc_weights = cycle_gan.model()
    #optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    #summary_op = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver(max_to_keep=1)

  with tf.Session(graph=graph) as sess:
    # sess.run(tf.global_variables_initializer())
    if FLAGS.load_model.strip() is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      best_acc = 2712
      # meta_graph_path = os.path.join(checkpoints_dir, '{}_best_model_{}.ckpt.meta'.format(FLAGS.load_model.strip(),best_acc))
      #epoch = int(meta_graph_path.split("_")[-1].split(".")[0])
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      # restore.restore(sess, './checkpoints/wgan4_sim_best_model_7543.ckpt')
      # restore.restore(sess, os.path.join(checkpoints_dir, '{}_best_model_{}.ckpt'.format(FLAGS.load_model.strip(),best_acc)))
      #step = int(meta_graph_path.split("-")[2].split(".")[0])
      #epoch = 5
    else:
      sess.run(tf.global_variables_initializer())
    
    if FLAGS.image_size == 40:
      id2label = {"ribosome": 0, "membrane": 1, "TRiC": 2, "proteasome_s": 3}
    else:
      id2label = {"31": 0, "33": 1, "35": 2, "43": 3, "69": 4, "72": 5, "73": 6}
    image_size = FLAGS.image_size
     # ribosome , membrane, TRiC, proteasome_s
    img_den, img_sub, den_label, max_imgnum = load_picdata(
        FLAGS.image_size, FLAGS.X, FLAGS.Y, FLAGS.noise, label=len(id2label))
    # img_den = np.load("/mnt/ssd1/home/v_xindi_wu/proj/dataset/bigdensitymap/400/datasets.npy")
    # img_den = preprocessz(img_den)


    x = img_den.reshape(-1, image_size, image_size, image_size, 1)
    y = img_sub.reshape(-1, image_size, image_size, image_size, 1)
    x_label = den_label

    #print(x[0,:])
    fake_subtomogram = {}
    fake_density = {}
    label2id = {}
    for idx , tag in id2label.items():
      fake_density[idx] = []
      fake_subtomogram[idx] = []
      label2id[tag] = idx

    fake_sub_slice = []

    for step in range(x.shape[0]//FLAGS.batch_size):
      # get previously generated images
      fake_y_val = sess.run(fake_y, feed_dict={cycle_gan.x: x[step, :].reshape(-1, image_size, image_size, image_size, 1)})
      # fake_y_val, fake_x_val = sess.run([fake_y, fake_x], feed_dict={cycle_gan.x: x[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size, :],
      #                                                               cycle_gan.y: y[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size, :]})
      fake_subtomogram[label2id[x_label[step]]].append(fake_y_val.reshape((image_size,image_size,image_size,1)))
      print("step: {}".format(step))
      #print(fake_y_val)
      # fake_subtomogram[label2id[int(x_label[step])]].append(fake_y_val.reshape((image_size,image_size,image_size,1)))
      #fake_density[label2id[int(sub_label[step])]].append(fake_x_val.reshape((image_size,image_size,image_size,1)))
      #print(tag_label[int(label[step])])

    with open("../result/{}/uncertainty_est_sub.pickle".format(checkpoints_dir.split("/")[-1]), "wb") as f:
      pickle.dump(fake_subtomogram, f)  # py3 load : encoding='iso-8859-1'
    # img_fake_sub = np.concatenate(fake_sub_slice, axis=0)
    # img_fake_sub = img_fake_sub.reshape((-1,image_size,image_size,image_size,1))
    # np.save("../result/{}/best_acc_fake_subtomogram_test".format(checkpoints_dir.split("/")[-1]),img_fake_sub)#1KP8
    # np.save("../result/{}/density_map_test".format(checkpoints_dir.split("/")[-1]),x)#1KP8
    # np.save("../result/{}/subtomogram_test".format(checkpoints_dir.split("/")[-1]),y)#subtomogram
    # np.save("../result/{}/density_map_label_test".format(checkpoints_dir.split("/")[-1]),x_label)
    # np.save("../result/{}/subtomogram_label_test".format(checkpoints_dir.split("/")[-1]),x_label)
    
    return

def main(unused_argv):
  uncertainty_est()
  # evaluate()

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES']='7'
  print(tf.test.is_gpu_available())
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
