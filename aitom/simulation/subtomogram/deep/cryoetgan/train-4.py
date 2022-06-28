# -*- coding=utf-8 -*-
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from model import CycleGAN

# from tensorflow.compat.v1 import keras
from datetime import datetime
import os
import logging
from utils import ImagePool

import numpy as np
#import pickle
import pickle
import keras

# import keras # pip3 install keras==2.2.5 and modify 
# File "/usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py", line 3508, in softmax    'return tf.nn.softmax(x, axis=axis)' to 'return tf.nn.softmax(x, axis)'
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
import sklearn
from sklearn.metrics import mean_squared_error
from prdc import compute_prdc
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from dataLoader import load_picdata, preprocessz, Toslice, read_data, get_picdata

FLAGS = tf.flags.FLAGS
# FLAGS = tf.compat.flags.Flag

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
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20190201-1553), default: None')  # ribosome , membrane, TRiC, proteasome_s
# Evaluation Config
tf.flags.DEFINE_string('classification_model', '../data/trainonsub.h5',
                       'the path that the checkpoints of model used to classify, it should be keras model, default: None')
tf.flags.DEFINE_integer('k_prdc', 5, 'the number of nearest neighbour when computing precision, recall, density, and coverage')
# ribosome , membrane, TRiC, proteasome_s
tf.flags.DEFINE_string('new_model', 'wgan4_sn_lr2e-4','new model name')
tf.flags.DEFINE_float('noise', 0, 'noise sigma')
tf.flags.DEFINE_integer('epoch', 50, 'epoch number, default: 256')
'''
def draw(data,name):
  x, y, z = np.where(data > 0.5)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, y, z, zdir='z', c='red')
  plt.savefig(name)
  plt.close()
'''

def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "../checkpoints/" + FLAGS.load_model.strip()
    print(checkpoints_dir)
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "../checkpoints/{}".format(FLAGS.new_model.strip())
    
    try:
      os.makedirs(checkpoints_dir)
      os.makedirs("../result/" + FLAGS.new_model.strip())
    except os.error:
      pass

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
    optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()
    saver_best = tf.train.Saver(max_to_keep=1)

  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      #meta_graph_path = './checkpoints/20190218-2122/model.ckpt-72.meta'
      epoch = int(meta_graph_path.split("-")[-1].split(".")[0])
      # epoch = 55
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      #restore.restore(sess, './checkpoints/20190218-2122/model.ckpt-72')
      #step = int(meta_graph_path.split("-")[2].split(".")[0])
      step = 1
      #epoch = 5
    else:
      sess.run(tf.global_variables_initializer())
      step = 1
      epoch = 0

    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

     # ribosome , membrane, TRiC, proteasome_s
    if (FLAGS.image_size==40):
      id2label = {"ribosome": 0, "membrane": 1, "TRiC": 2, "proteasome_s": 3}
    else:
      id2label = {"31": 0, "33": 1, "35": 2, "43": 3, "69": 4, "72": 5,"73":6}

    img_den, img_sub, den_label, max_imgnum = load_picdata(
        FLAGS.image_size, FLAGS.X, FLAGS.Y, FLAGS.noise, label=len(id2label))
    sub_label = den_label.copy()
    image_size = FLAGS.image_size
    

    x = img_den.reshape(-1, image_size, image_size, image_size, 1)
    y = img_sub.reshape(-1, image_size, image_size, image_size, 1)
    x_label = den_label

    img_fake_sub = []
    img_fake_den = None
    acc_score = []
    best_score = 0
    if not (FLAGS.classification_model == "0"):
      model = load_model(FLAGS.classification_model)
      print("cls Model Loaded")

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)

      while True:
        # get previously generated images
        fake_y_val, fake_x_val = sess.run([fake_y, fake_x], feed_dict={cycle_gan.x: x[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size, :],
                                                                       cycle_gan.y: y[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size, :]})

        if FLAGS.use_wloss:
          # train
          _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary, cycle_loss_val, G_gan_loss_val, F_gan_loss_val,_ = (
                sess.run(
                    [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op, cycle_loss, G_gan_loss, F_gan_loss, clip_disc_weights],
                    feed_dict={cycle_gan.x:x[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size,:],
                              cycle_gan.y:y[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size,:],
                              cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                              cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
                )
          )
        else:
            # train
          _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary, cycle_loss_val, G_gan_loss_val, F_gan_loss_val = (
                sess.run(
                    [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op, cycle_loss, G_gan_loss, F_gan_loss],
                    feed_dict={cycle_gan.x:x[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size,:],
                              cycle_gan.y:y[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size,:],
                              cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                              cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
                )
          )
        # _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary, cycle_loss_val, G_gan_loss_val, F_gan_loss_val = (
        #       sess.run(
        #           [optimizers, G_loss, D_Y_loss, F_loss,
        #            D_X_loss, summary_op, cycle_loss, G_gan_loss, F_gan_loss],
        #           feed_dict={cycle_gan.x:x[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size,:],
        #                     cycle_gan.y:y[(step-1)*FLAGS.batch_size:step*FLAGS.batch_size,:],
        #                     cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
        #                     cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
        #       )
        # )

        train_writer.add_summary(summary, step)
        train_writer.flush()

        if step % 1 == 0:
          logging.info('-----------Epoch %d:-------------' % epoch)
          logging.info('-----------Step %d:-------------' % step)
          logging.info('  G_loss   : {}'.format(G_loss_val))
          logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
          logging.info('  F_loss   : {}'.format(F_loss_val))
          logging.info('  D_X_loss : {}'.format(D_X_loss_val))
          logging.info('  cycle_loss : {}'.format(cycle_loss_val))
          logging.info('  G_gan_loss : {}'.format(G_gan_loss_val))
          logging.info('  F_gan_loss : {}'.format(F_gan_loss_val))


          '''
          for fake_id in range(FLAGS.batch_size):
            fake_y_plt = fake_y[fake_id,:].reshape((40,40,40))
            fake_x_plt = fake_x[fake_id,:].reshape((40,40,40))
            draw(fake_x_plt, name="result/fake_x_"+str(fake_id))
            draw(fake_y_plt, name="result/fake_y_"+str(fake_id))
          '''
        if epoch%1 == 0:

          #img_fake_den.append(fake_x_val)
          # if img_fake_sub is None:
          #   img_fake_sub = fake_y_val.reshape((-1,image_size,image_size,image_size,1))
          # else:
          #   #img_fake_sub.append(fake_y_val)
          #   img_fake_sub = np.concatenate((img_fake_sub,fake_y_val.reshape((-1,image_size,image_size,image_size,1))),axis=0)
          img_fake_sub.append(fake_y_val.reshape((-1,image_size,image_size,image_size,1)))
          #np.save("./result/1KP8_"+str(step),fake_x_val)
          #np.save("./result/2AWB_"+str(step),fake_y_val)

          if (FLAGS.classification_model != "0") and (step % int(max_imgnum/FLAGS.batch_size) == 0):
            #img_fake_den = np.array(img_fake_den)
            img_fake_sub = np.concatenate(img_fake_sub, axis=0)
            img_fake_sub = np.array(img_fake_sub).reshape((-1,image_size,image_size,image_size,1))
            labels_test = np_utils.to_categorical(x_label, 4)
            scores = model.evaluate(img_fake_sub, labels_test)
            acc_score.append(scores[1])

            if best_score < scores[1]:
              best_score = scores[1]
              save_path = saver_best.save(sess, os.path.join(checkpoints_dir, checkpoints_dir.split("/")[-1]+"_best_model_{}.ckpt".format(int(best_score*10000))), global_step=epoch)
              logging.info("Best model saved in file: %s" % save_path)

              # metrics = compute_prdc(real_features=y.reshape((x.shape[0],-1)),
              #           fake_features=img_fake_sub.reshape((x.shape[0],-1)),
              #           nearest_k=FLAGS.k_prdc)
              
              # compute psnr
              # mse = mean_squared_error(y.reshape((img_fake_sub.shape[0],-1)), img_fake_sub.reshape((img_fake_sub.shape[0],-1)))
              # psnr = 10. * np.log10(1. / mse)
              # with open(FLAGS.new_model.strip()+"acc_result.pickle", "wb") as f:
              #   pickle.dump(acc_score,f)
              with open(os.path.join(checkpoints_dir, checkpoints_dir.split("/")[-1]+"_best_model_metrics.json"),"w", encoding='utf-8') as f:
                json.dump({"accuracy":best_score}, f, indent=4)

              np.save("../result/{}/best_acc_fake_subtomogram".format(checkpoints_dir.split("/")[-1]),img_fake_sub)#1KP8
              np.save("../result/{}/density_map".format(checkpoints_dir.split("/")[-1]),x)#1KP8
              np.save("../result/{}/subtomogram".format(checkpoints_dir.split("/")[-1]),y)#subtomogram
              np.save("../result/{}/density_map_label".format(checkpoints_dir.split("/")[-1]),x_label)
              np.save("../result/{}/subtomogram_label".format(checkpoints_dir.split("/")[-1]),sub_label)


            plt.title('accuracy')
            plt.plot(list(range(len(acc_score))), acc_score, color="red")

            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.savefig(os.path.join(checkpoints_dir, checkpoints_dir.split("/")[-1]+'.png'), dpi=200)

            print("Epoch {}".format(epoch))
            print("Accuracy in real-sub classification :{}".format(scores[1]))
            img_fake_sub = []
            #img_fake_den = []

        if step % int(max_imgnum/FLAGS.batch_size) == 0:
          if epoch%5 == 0:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=epoch)
            logging.info("Model saved in file: %s" % save_path)
          #save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=epoch)
          #logging.info("Model saved in file: %s" % save_path)
          epoch = epoch + 1
          step = 0

          den_index = np.arange(img_den.shape[0])
          np.random.shuffle(den_index)
          img_den = img_den[den_index,:]
          den_label = den_label[den_index]

          sub_index = np.arange(img_sub.shape[0])
          np.random.shuffle(sub_index)
          img_sub = img_sub[sub_index, :]
          sub_label = sub_label[sub_index]

          x = img_den[0:max_imgnum,:].reshape(-1, image_size, image_size, image_size, 1)
          y = img_sub[0:max_imgnum,:].reshape(-1, image_size, image_size, image_size, 1)
          x_label = den_label[0:max_imgnum]
          
          logging.info("shuffle image")
          if FLAGS.epoch == epoch:
            print("end train")
            break

        step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      #coord.request_stop()

    finally:
      print("training stop")
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=epoch)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      #coord.request_stop()
      #coord.join(threads)


def main(unused_argv):
  train()

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES']='5'
  print(tf.test.is_gpu_available())
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
