import tensorflow as tf
from tensorflow.keras import backend as K
from keras.engine.topology import Layer


def _featurenorm(feature):
    epsilon = 1e-6
    norm = tf.pow(tf.reduce_sum(tf.pow(feature, 2), 1) + epsilon, 0.5)
    norm = tf.expand_dims(norm, 1)
    norm = tf.tile(norm, [1, feature.get_shape().as_list()[1], 1, 1, 1])
    norm = tf.divide(feature, norm)

    return norm


class FeatureL2Norm(Layer):
    """
    Normalizing features using l2 norm
    Modified from https://github.com/jaehyunnn/cnngeometric_tensorflow

    References
    ----------
    [1]  Convolutional neural network architecture for geometric matching, Ignacio Rocco, et al.
    """

    def __init__(self, **kwargs):
        super(FeatureL2Norm, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        length = input_shapes[1]
        height = input_shapes[2]
        width = input_shapes[3]
        num_channels = input_shapes[4]
        return None, length, height, width, num_channels

    def call(self, feature):
        output = _featurenorm(feature=feature)
        return output

    def get_config(self):
        base_config = super(FeatureL2Norm, self).get_config()
        return dict(list(base_config.items()))

