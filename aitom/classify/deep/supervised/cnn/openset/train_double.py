from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import os
import time
import sys
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.contrib import slim
import tensorflow.contrib.data as tf_data
from collections import Counter
import numpy as np
import importlib
import itertools
import tensorflow.contrib.slim as slim
import argparse
from . import utils
from . import sphere_network as network
import pdb
from tensorflow.python.ops import data_flow_ops

debug = False
softmax_ind = 0


def _from_tensor_slices(tensors_x, tensors_y):
    # return TensorSliceDataset((tensors_x,tensors_y))
    return tf.data.Dataset.from_tensor_slices((tensors_x, tensors_y))


def get_center_loss(features, labels, alpha, num_classes):
    """
    Arguments:
        features: Tensor,shape [batch_size, feature_length].
        labels: Tensor,shape [batch_size].#not the one hot label
        alpha:  center upgrade learning rate
        num_classes: how many classes.

    Return：
        loss: Tensor,
        centers: Tensor
        centers_update_op:
    """
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0),
                              trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    loss = tf.nn.l2_loss(features - centers_batch)
    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    # need to update after every epoch, the key is to update the center of the classes.

    return loss, centers, centers_update_op


def read_npy_file(item):
    data = np.load(item)


def main_train(args):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    # Create the log directory if it doesn't exist
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    # Create the model directory if it doesn't exist
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    # Write arguments to a text file
    utils.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    utils.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = utils.dataset_from_list(args.train_data_dir,
                                        args.train_list_dir) # class objects in a list

    # ----------------------class definition-------------------------------------
    # class ImageClass:
    #     """Stores the paths to images for a given class"""
    #     def __init__(self, name, image_paths):
    #         self.name = name
    #         self.image_paths = image_paths
    #
    #     def __str__(self):
    #         return self.name + ', ' + str(len(self.image_paths)) + ' images'
    #
    #     def __len__(self):
    #         return len(self.image_paths)

    nrof_classes = len(train_set)
    print('nrof_classes: ', nrof_classes)
    image_list, label_list = utils.get_image_paths_and_labels(train_set)
    # label is in the form scalar.
    print('total images: ', len(image_list))
    image_list = np.array(image_list)
    label_list = np.array(label_list, dtype=np.int32)
    dataset_size = len(image_list)
    single_batch_size = args.class_per_batch * args.images_per_class
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    def _sample_people_softmax(x):
        """loading the images in batches"""
        global softmax_ind
        if softmax_ind >= dataset_size:
            np.random.shuffle(indices)
            softmax_ind = 0
        true_num_batch = min(single_batch_size, dataset_size - softmax_ind)

        sample_paths = image_list[indices[softmax_ind:softmax_ind + true_num_batch]]
        sample_images = []

        for item in sample_paths:
            sample_images.append(np.load(str(item)))
            # print(item)
        # print(type(sample_paths[0]))
        sample_labels = label_list[indices[softmax_ind:softmax_ind + true_num_batch]]
        softmax_ind += true_num_batch
        return (np.expand_dims(np.array(sample_images, dtype=np.float32),
                               axis=4), np.array(sample_labels, dtype=np.int32))

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        # the image is generated by sequence
        with tf.device("/cpu:0"):
            softmax_dataset = tf.data.Dataset.range(args.epoch_size * args.max_nrof_epochs)
            softmax_dataset = softmax_dataset.map(
                lambda x: tf.py_func(_sample_people_softmax, [x], [tf.float32, tf.int32]))
            softmax_dataset = softmax_dataset.flat_map(_from_tensor_slices)
            softmax_dataset = softmax_dataset.batch(single_batch_size)
            softmax_iterator = softmax_dataset.make_initializable_iterator()
            softmax_next_element = softmax_iterator.get_next()
            softmax_next_element[0].set_shape(
                (single_batch_size, args.image_height, args.image_width, args.image_width, 1))
            softmax_next_element[1].set_shape(single_batch_size)
            batch_image_split = softmax_next_element[0]
            # batch_image_split = tf.expand_dims(batch_image_split, axis = 4)
            batch_label_split = softmax_next_element[1]

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder,
                                                   global_step,
                                                   args.learning_rate_decay_epochs *
                                                   args.epoch_size,
                                                   args.learning_rate_decay_factor,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        print('Using optimizer: {}'.format(args.optimizer))
        if args.optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif args.optimizer == 'SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif args.optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        elif args.optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        else:
            raise Exception("Not supported optimizer: {}".format(args.optimizer))

        losses = {}
        with slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0"):
            with tf.variable_scope(tf.get_variable_scope()) as var_scope:
                reuse = False

                if args.network == 'sphere_network':

                    prelogits = network.infer(batch_image_split, args.embedding_size)
                else:
                    raise Exception("Not supported network: {}".format(args.network))

                if args.fc_bn:
                    prelogits = slim.batch_norm(prelogits,
                                                is_training=True,
                                                decay=0.997,
                                                epsilon=1e-5,
                                                scale=True,
                                                updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                reuse=reuse,
                                                scope='softmax_bn')

                if args.loss_type == 'softmax':
                    cross_entropy_mean = utils.softmax_loss(prelogits, batch_label_split,
                                                            len(train_set), 1.0, reuse)
                    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    loss = cross_entropy_mean + args.weight_decay * tf.add_n(regularization_losses)
                    print('************************' + ' Computing the softmax loss')
                    losses['total_loss'] = cross_entropy_mean
                    losses['total_reg'] = args.weight_decay * tf.add_n(regularization_losses)

                elif args.loss_type == 'lmcl':
                    label_reshape = tf.reshape(batch_label_split, [single_batch_size])
                    label_reshape = tf.cast(label_reshape, tf.int64)
                    coco_loss = utils.cos_loss(prelogits,
                                               label_reshape,
                                               len(train_set),
                                               reuse,
                                               alpha=args.alpha,
                                               scale=args.scale)
                    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    loss = coco_loss + args.weight_decay * tf.add_n(regularization_losses)
                    print('************************' + ' Computing the lmcl loss')
                    losses['total_loss'] = coco_loss
                    losses['total_reg'] = args.weight_decay * tf.add_n(regularization_losses)

                elif args.loss_type == 'center':
                    # center loss
                    center_loss, centers, centers_update_op = get_center_loss(
                        prelogits, label_reshape, args.center_loss_alfa, args.num_class_train)
                    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    loss = center_loss + args.weight_decay * tf.add_n(regularization_losses)
                    print('************************' + ' Computing the center loss')
                    losses['total_loss'] = center_loss
                    losses['total_reg'] = args.weight_decay * tf.add_n(regularization_losses)

                elif args.loss_type == 'lmccl':
                    cross_entropy_mean = utils.softmax_loss(prelogits, batch_label_split,
                                                            len(train_set), 1.0, reuse)
                    label_reshape = tf.reshape(batch_label_split, [single_batch_size])
                    label_reshape = tf.cast(label_reshape, tf.int64)
                    coco_loss = utils.cos_loss(prelogits,
                                               label_reshape,
                                               len(train_set),
                                               reuse,
                                               alpha=args.alpha,
                                               scale=args.scale)
                    center_loss, centers, centers_update_op = get_center_loss(
                        prelogits, label_reshape, args.center_loss_alfa, args.num_class_train)
                    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    reg_loss = args.weight_decay * tf.add_n(regularization_losses)
                    loss = coco_loss + reg_loss + args.center_weighting * center_loss + cross_entropy_mean
                    losses['total_loss_center'] = args.center_weighting * center_loss
                    losses['total_loss_lmcl'] = coco_loss
                    losses['total_loss_softmax'] = cross_entropy_mean
                    losses['total_reg'] = reg_loss

        grads = opt.compute_gradients(loss,
                                      tf.trainable_variables(),
                                      colocate_gradients_with_ops=True)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # used for updating the centers in the center loss.
        if args.loss_type == 'lmccl' or args.loss_type == 'center':
            with tf.control_dependencies([centers_update_op]):
                with tf.control_dependencies(update_ops):
                    train_op = tf.group(apply_gradient_op)
        else:
            with tf.control_dependencies(update_ops):
                train_op = tf.group(apply_gradient_op)

        save_vars = [
            var for var in tf.global_variables()
            if 'Adagrad' not in var.name and 'global_step' not in var.name
        ]
        saver = tf.train.Saver(save_vars, max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

        # sess.run(iterator.initializer)
        sess.run(softmax_iterator.initializer)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                if debug:
                    debug_train(args, sess, train_set, epoch, image_batch_gather, enqueue_op,
                                batch_size_placeholder, image_batch_split, image_paths_split,
                                num_per_class_split, image_paths_placeholder,
                                image_paths_split_placeholder, labels_placeholder, labels_batch,
                                num_per_class_placeholder, num_per_class_split_placeholder,
                                len(gpus))
                # Train for one epoch
                if args.loss_type == 'lmccl' or args.loss_type == 'center':
                    train_contain_center(args, sess, epoch, learning_rate_placeholder,
                                         phase_train_placeholder, global_step, losses, train_op,
                                         summary_op, summary_writer, '', centers_update_op)
                else:
                    train(args, sess, epoch, learning_rate_placeholder, phase_train_placeholder,
                          global_step, losses, train_op, summary_op, summary_writer, '')
                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
    return model_dir


def train_contain_center(args, sess, epoch, learning_rate_placeholder, phase_train_placeholder,
                         global_step, loss, train_op, summary_op, summary_writer,
                         learning_rate_schedule_file, centers_update_op):
    batch_number = 0
    lr = args.learning_rate
    while batch_number < args.epoch_size:
        start_time = time.time()
        print('Running forward pass on sampled images: ', end='')
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True}
        start_time = time.time()
        # for double_loss
        if args.loss_type == 'lmccl':
            total_err_center, total_err_lmcl, reg_err, total_err_softmax, _, step, _ = sess.run(
                [
                    loss['total_loss_center'], loss['total_loss_lmcl'], loss['total_reg'],
                    loss['total_loss_softmax'], train_op, global_step, centers_update_op
                ],
                feed_dict=feed_dict)
            duration = time.time() - start_time
            print(
                'Epoch: [%d][%d/%d]\tTime %.3f\tTotal center Loss %2.3f\tTotal lmcl Loss %2.3f'
                '\tReg Loss %2.3f\tTotal softmax Loss %2.3f, lr %2.5f'
                % (epoch, batch_number + 1, args.epoch_size, duration, total_err_center,
                   total_err_lmcl, reg_err, total_err_softmax, lr))
        else:
            total_err, reg_err, _, step, _ = sess.run(
                [loss['total_loss'], loss['total_reg'], train_op, global_step, centers_update_op],
                feed_dict=feed_dict)

            duration = time.time() - start_time
            print(
                'Epoch: [%d][%d/%d]\tTime %.3f\tTotal center Loss %2.3f\tReg Loss %2.3f, lr %2.5f' %
                (epoch, batch_number + 1, args.epoch_size, duration, total_err, reg_err, lr))

        batch_number += 1
    return step


def train(args, sess, epoch, learning_rate_placeholder, phase_train_placeholder, global_step, loss,
          train_op, summary_op, summary_writer, learning_rate_schedule_file):
    batch_number = 0
    lr = args.learning_rate
    while batch_number < args.epoch_size:
        start_time = time.time()
        print('Running forward pass on sampled images: ', end='')
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True}
        start_time = time.time()
        # for cosface loss.
        total_err, reg_err, _, step = sess.run(
            [loss['total_loss'], loss['total_reg'], train_op, global_step], feed_dict=feed_dict)
        duration = time.time() - start_time
        if arg.loss_type == 'lmcl':
            print('Epoch: [%d][%d/%d]\tTime %.3f\tTotal lmcl Loss %2.3f\tReg Loss %2.3f, lr %2.5f' %
                  (epoch, batch_number + 1, args.epoch_size, duration, total_err, reg_err, lr))
        else:
            print(
                'Epoch: [%d][%d/%d]\tTime %.3f\tTotal softmax Loss %2.3f\tReg Loss %2.3f, lr %2.5f'
                % (epoch, batch_number + 1, args.epoch_size, duration, total_err, reg_err, lr))
        batch_number += 1
    return step


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
