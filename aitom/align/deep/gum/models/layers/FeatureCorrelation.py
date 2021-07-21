import tensorflow as tf
from tensorflow.keras import backend as K
from keras.engine.topology import Layer


class FeatureCorrelation(Layer):
    """
    Performs feature correlation as a keras layer
    Modified from https://github.com/jaehyunnn/cnngeometric_tensorflow

    References
    ----------
    [1]  Convolutional neural network architecture for geometric matching, Ignacio Rocco, et al.
    """
    def __init__(self, **kwargs):
        super(FeatureCorrelation, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        length = input_shapes[1][1]
        height = input_shapes[1][2]
        width = input_shapes[1][3]
        length0 = input_shapes[0][1]
        height0 = input_shapes[0][2]
        width0 = input_shapes[0][3]
        num_channels = length0 * height0 * width0
        return None, length, height, width, num_channels

    def call(self, tensors):
        f_A, f_B = tensors
        output = self._featurecorrelation(f_A=f_A, f_B=f_B)
        return output

    def get_config(self):
        base_config = super(FeatureCorrelation, self).get_config()

        return dict(list(base_config.items()))

    def _featurecorrelation(self, f_A, f_B):
        b, l0, h0, w0, c = f_A.get_shape().as_list()
        b, l, h, w, c = f_B.get_shape().as_list()

        f_A = tf.transpose(f_A, [0, 3, 2, 1, 4])
        f_A = tf.reshape(f_A, [-1, l0*h0*w0, c])

        f_B = tf.reshape(f_B, [-1, l*h*w, c])
        f_B = tf.transpose(f_B, [0, 2, 1])

        f_mul = tf.matmul(f_A, f_B)
        correlation_tensor = tf.reshape(f_mul, [-1, l, h, w, l0*h0*w0])

        return correlation_tensor

