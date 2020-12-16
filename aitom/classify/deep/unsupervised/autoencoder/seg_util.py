"""
Authors of the code: Xiangrui Zeng, Min Xu
License: ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

Reference:
Zeng X, Leung M, Zeev-Ben-Mordehai T, Xu M. A convolutional autoencoder approach for mining features in cellular electron cryo-tomograms and weakly supervised coarse segmentation. Journal of Structural Biology (2017) doi:10.1016/j.jsb.2017.12.015

Please cite the above paper when this code is used or adapted for your research.
"""

"""
convolutional layer block

for the use of batch normalization, see
http://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
best model [GoogLeNet128_BN_lim0606] actually has the BN layer BEFORE the ReLU

https://github.com/titu1994/Inception-v4/blob/master/inception_v4.py
"""

import keras.backend as K
import tensorflow as tf
import functools


def conv_block(x,
               nb_filter,
               nb0,
               nb1,
               nb2,
               border_mode='same',
               subsample=(1, 1, 1),
               bias=True,
               batch_norm=False):
    from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, UpSampling3D, Reshape, Flatten, Activation
    from keras.layers.normalization import BatchNormalization

    from keras import backend as K
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Convolution3D(nb_filter,
                      nb0,
                      nb1,
                      nb2,
                      subsample=subsample,
                      border_mode=border_mode,
                      bias=bias)(x)
    if batch_norm:
        assert not bias
        x = BatchNormalization(axis=channel_axis)(x)
    else:
        assert bias

    x = Activation('relu')(x)

    return x


def weighted_mean_squared_error(y_true, y_pred, weights):
    """
    weighted mean square error, for weight balancing, see
    https://github.com/fchollet/keras/issues/2115
    https://github.com/fchollet/keras/blob/master/keras/losses.py
    """
    mask = K.zeros_like(y_true)
    for lbl in weights:
        # An adhoc way to solve the problem of: tensor object does not support item assignment
        mask += K.cast(K.equal(y_true, lbl), dtype='float32') * (K.zeros_like(y_true) + weights[lbl])
    return K.mean(K.square(y_pred - y_true) * mask, axis=-1)


def w_binary_crossentropy(output, target, weights):
    """
    Define a weighted binary_crossentropy loss
    """
    _EPSILON = 10e-8
    _FLOATX = 'float32'

    output = tf.clip_by_value(output, tf.cast(_EPSILON, dtype=_FLOATX),
                              tf.cast(1. - _EPSILON, dtype=_FLOATX))
    output = tf.log(output / (1 - output))
    return tf.nn.weighted_cross_entropy_with_logits(output, target, weights)


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)

    functools.update_wrapper(partial_func, func)

    return partial_func
