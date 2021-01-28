import tensorflow as tf
from tensorflow.spectral import dct, idct

from tensorflow.keras import backend as K
from keras.engine.topology import Layer


class SpectralPooling(Layer):
    """
    Performs spectral pooling and filtering using DCT as a keras layer
    """

    def __init__(self, output_size, truncation, homomorphic=False, **kwargs):
        self.output_size = output_size
        self.truncation = truncation
        self.homomorphic = homomorphic
        super(SpectralPooling, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        length, height, width = self.output_size
        num_channels = input_shapes[-1]
        return None, length, height, width, num_channels

    def call(self, tensors, mask=None):
        if self.homomorphic:
            tensors = K.log(tensors)

        x_dct = self._dct3D(tensors)
        x_crop = self._cropping3D(x_dct)
        x_idct = self._idct3D(x_crop)

        if self.homomorphic:
            x_idct = K.exp(x_idct)

        return x_idct

    def get_config(self):
        config = {
            'output_size': self.output_size,
            'truncation': self.truncation,
            'homomorphic': self.homomorphic}
        base_config = super(SpectralPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _dct3D(self, x):
        x_perm = tf.transpose(x, perm=[0, 4, 1, 2, 3])
        output = tf.transpose(dct(tf.transpose(dct(
            tf.transpose(dct(x_perm, 2, norm='ortho'), perm=[0, 1, 2, 4, 3]),
            2, norm='ortho'), perm=[0, 1, 3, 4, 2]), 2, norm='ortho'), perm=[0, 4, 3, 2, 1])

        return output

    def _idct3D(self, x):
        x_perm = tf.transpose(x, perm=[0, 4, 3, 2, 1])
        output = tf.transpose(idct(tf.transpose(idct(
            tf.transpose(idct(x_perm, 2, norm='ortho'), perm=[0, 1, 4, 2, 3]),
            2, norm='ortho'), perm=[0, 1, 2, 4, 3]), 2, norm='ortho'), perm=[0, 2, 3, 4, 1])

        return output

    def _cropping3D(self, x):
        x_trunc = x[:, :self.truncation[0], :self.truncation[1], :self.truncation[2], :]
        paddings = tf.constant([[0, 0],
                                [0, self.output_size[0] - self.truncation[0]],
                                [0, self.output_size[1] - self.truncation[1]],
                                [0, self.output_size[2] - self.truncation[2]],
                                [0, 0]])

        output = tf.pad(x_trunc, paddings, "CONSTANT")

        return output
