import keras.backend as K
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Reshape, Lambda
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D, Conv3D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import Flatten
from keras_contrib.layers.convolutional import Deconvolution3D

from keras.engine.topology import Layer

kernel_size = (4, 4, 4)


def initNormal(shape, dtype=None):
    my_init = initializers.RandomNormal(mean=0., stddev=0.02, seed=None)
    return my_init(shape)


def initConstant(shape, dtype=None):
    my_init = initializers.Constant(value=0.0)
    return my_init(shape)


class MinibatchDiscrimination(Layer):
    def __init__(self, num_kernels, kernel_dim, **kwargs):
        self.nb_kernels = num_kernels
        self.kernel_dim = kernel_dim
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.nb_kernels, input_shape[1], self.kernel_dim),
                                      initializer=initNormal,
                                      trainable=True)
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x):
        activation = K.reshape(K.dot(x, self.kernel), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + self.nb_kernels

    # WGAN generator


def w_generator_model(GEN_INPUT_DIM):
    leaky_grad = 0.2
    model = Sequential()

    model.add(Dense(input_dim=GEN_INPUT_DIM, units=4 * 4 * 4 * 256,
                    kernel_initializer=initNormal,
                    bias_initializer=initConstant))

    model.add(Reshape((4, 4, 4, 256), input_shape=(4 * 4 * 4 * 256,)))

    # Input: (,4,4,4,256)
    model.add(Deconvolution3D(256, kernel_size=kernel_size, padding='same',
                              output_shape=(None, 4, 4, 4, 256), strides=(1, 1, 1),
                              kernel_initializer=initNormal,
                              bias_initializer=initConstant))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))

    model.add(Deconvolution3D(128, kernel_size=kernel_size, padding='same',
                              output_shape=(None, 8, 8, 8, 128), strides=(2, 2, 2),
                              kernel_initializer=initNormal,
                              bias_initializer=initConstant))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))

    model.add(Deconvolution3D(64, kernel_size=kernel_size, padding='same',
                              output_shape=(None, 16, 16, 16, 64), strides=(2, 2, 2),
                              kernel_initializer=initNormal,
                              bias_initializer=initConstant))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))

    model.add(Deconvolution3D(1, kernel_size=kernel_size, padding='same',
                              output_shape=(None, 32, 32, 32, 1), strides=(2, 2, 2),
                              kernel_initializer=initNormal,
                              bias_initializer=initConstant))
    model.add(Activation('tanh'))
    return model


def w_discriminator_model(input_shape):
    """WGAN Discriminator"""
    leaky_grad = 0.2
    dropout = 0.0
    model = Sequential()

    # Input: (,32,32,32,1)
    # Output: (,16,16,16,64)
    model.add(Conv3D(64, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     input_shape=input_shape, kernel_initializer=initNormal,
                     bias_initializer=initConstant))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,16,16,16,64)
    # Output: (,8,8,8,128) 
    model.add(Conv3D(128, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     kernel_initializer=initNormal,
                     bias_initializer=initConstant))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,8,8,8,128)
    # Output: (,4,4,4,256)
    model.add(Conv3D(256, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     kernel_initializer=initNormal,
                     bias_initializer=initConstant))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,4,4,4,256)
    # Output: (,4,4,4,256)
    model.add(Conv3D(256, kernel_size=kernel_size, padding='same', strides=(1, 1, 1),
                     kernel_initializer=initNormal,
                     bias_initializer=initConstant))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # (4*4*4*256,)
    model.add(Flatten())

    model.add(MinibatchDiscrimination(num_kernels=64, kernel_dim=4))

    model.add(Dense(1, kernel_initializer=initNormal))
    return model


def regressor_model(input_shape, INPUT_DIM, d_model):
    """
    Regressor model
    Note the weights are hardcoded by layers, so do not change architecture!
    """
    model = Sequential()
    leaky_grad = 0.2

    # Input: (,32,32,32,1)
    # Output: (,16,16,16,64)
    model.add(Conv3D(64, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     input_shape=input_shape, weights=d_model.layers[0].get_weights()))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))

    # Input: (,16,16,16,64)
    # Output: (,8,8,8,128) 
    model.add(Conv3D(128, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     weights=d_model.layers[3].get_weights()))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))

    # Input: (,8,8,8,128)
    # Output: (,4,4,4,256)
    model.add(Conv3D(256, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     weights=d_model.layers[6].get_weights()))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))

    # Input: (,4,4,4,256)
    # Output: (,4,4,4,256)
    model.add(Conv3D(256, kernel_size=kernel_size, padding='same', strides=(1, 1, 1),
                     weights=d_model.layers[9].get_weights()))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))
    # (4*4*4*256,)
    model.add(Flatten())

    model.add(Dense(INPUT_DIM, kernel_initializer='he_normal'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def large_generator_model(GEN_INPUT_DIM):
    """Generator for shapes of size 64^3"""
    leaky_grad = 0.2
    model = Sequential()

    model.add(Dense(input_dim=GEN_INPUT_DIM, units=4 * 4 * 4 * 256,
                    kernel_initializer=initNormal,
                    bias_initializer=initConstant))

    model.add(Reshape((4, 4, 4, 256), input_shape=(4 * 4 * 4 * 256,)))

    # Input: (,4,4,4,256)
    model.add(Deconvolution3D(128, kernel_size=kernel_size, padding='same',
                              output_shape=(None, 8, 8, 8, 128),
                              strides=(2, 2, 2),
                              kernel_initializer=initNormal,
                              bias_initializer=initConstant))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))

    model.add(Deconvolution3D(64, kernel_size=kernel_size, padding='same',
                              output_shape=(None, 16, 16, 16, 64),
                              strides=(2, 2, 2),
                              kernel_initializer=initNormal,
                              bias_initializer=initConstant))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))

    model.add(Deconvolution3D(32, kernel_size=kernel_size, padding='same',
                              output_shape=(None, 32, 32, 32, 32),
                              strides=(2, 2, 2),
                              kernel_initializer=initNormal,
                              bias_initializer=initConstant))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))

    model.add(Deconvolution3D(1, kernel_size=kernel_size, padding='same',
                              output_shape=(None, 64, 64, 64, 1),
                              strides=(2, 2, 2),
                              kernel_initializer=initNormal,
                              bias_initializer=initConstant))
    model.add(Activation('tanh'))
    return model


def large_discriminator_model(input_shape):
    """Discriminator for 64^3"""
    leaky_grad = 0.2
    dropout = 0.0
    model = Sequential()

    # Input: (,64,64,64,1)
    # Output: (,32,32,32,32)
    model.add(Conv3D(32, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     input_shape=input_shape, kernel_initializer=initNormal,
                     bias_initializer=initConstant))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,32,32,32,32)
    # Output: (,16,16,16,64) 
    model.add(Conv3D(64, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     kernel_initializer=initNormal,
                     bias_initializer=initConstant))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,16,16,16,64)
    # Output: (,8,8,8,128)
    model.add(Conv3D(128, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     kernel_initializer=initNormal,
                     bias_initializer=initConstant))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,8,8,8,128)
    # Output: (,4,4,4,256)
    model.add(Conv3D(256, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     kernel_initializer=initNormal,
                     bias_initializer=initConstant))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # (4*4*4*256,)
    model.add(Flatten())

    model.add(MinibatchDiscrimination(num_kernels=64, kernel_dim=4))

    model.add(Dense(1, kernel_initializer=initNormal))
    return model


def l2_discriminator_model(input_shape, d_model):
    """
    note the weights are hardcoded by layers, so do not change architecture!
    discriminator that exposes the last layer, for calculating L2 distances in the latent space.
    """
    leaky_grad = 0.2
    dropout = 0.0
    model = Sequential()

    # Input: (,32,32,32,1)
    # Output: (,16,16,16,64)
    model.add(Conv3D(32, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     input_shape=input_shape, weights=d_model.layers[0].get_weights()))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,16,16,16,64)
    # Output: (,8,8,8,128) 
    model.add(Conv3D(64, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     weights=d_model.layers[3].get_weights()))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,8,8,8,128)
    # Output: (,4,4,4,256)
    model.add(Conv3D(128, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     weights=d_model.layers[6].get_weights()))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,4,4,4,256)
    # Output: (,4,4,4,256)
    model.add(Conv3D(256, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     weights=d_model.layers[9].get_weights()))
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # (4*4*4*256,)
    model.add(Flatten())

    return model


def large_regressor_model(input_shape, INPUT_DIM, d_model):
    """Regressor for 64^3"""
    model = Sequential()
    leaky_grad = 0.2
    dropout = 0.0
    BATCHNORM = False

    # Input: (,32,32,32,1)
    # Output: (,16,16,16,64)
    model.add(Conv3D(32, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     input_shape=input_shape, weights=d_model.layers[0].get_weights()))
    if BATCHNORM:
        model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,16,16,16,64)
    # Output: (,8,8,8,128) 
    model.add(Conv3D(64, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     weights=d_model.layers[3].get_weights()))
    if BATCHNORM:
        model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,8,8,8,128)
    # Output: (,4,4,4,256)
    model.add(Conv3D(128, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     weights=d_model.layers[6].get_weights()))
    if BATCHNORM:
        model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))

    # Input: (,4,4,4,256)
    # Output: (,4,4,4,256)
    model.add(Conv3D(256, kernel_size=kernel_size, padding='same', strides=(2, 2, 2),
                     weights=d_model.layers[9].get_weights()))
    model.add(BatchNormalization())
    model.add(LeakyReLU(leaky_grad))
    model.add(Dropout(dropout))
    # (4*4*4*256,)
    model.add(Flatten())

    model.add(Dense(INPUT_DIM, kernel_initializer='he_normal'))
    return model
