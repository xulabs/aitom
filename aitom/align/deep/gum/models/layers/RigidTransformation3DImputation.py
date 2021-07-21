from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from keras.engine.topology import Layer

from tensorflow.spectral import fft, ifft

from tensorflow.python import roll as _roll
from tensorflow.python.framework import ops

from tensorflow.keras.backend import ndim


class RigidTransformation3DImputation(Layer):
    """
    Performs bilinear interpolation as a keras layer
    for 3D rigid body transformation
    Modified from https://github.com/Ryo-Ito/spatial_transformer_network

    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    """

    # padding_method can currently either be "fill" or "replicate"
    def __init__(self, output_size, padding_method="fill", **kwargs):
        self.output_size = output_size
        self.padding_method = padding_method
        super(RigidTransformation3DImputation, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        length, height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return [(None, length, height, width, num_channels)] * 3

    # Rescales theta note theta is in range [0,1]
    # which is not symmetric about 0
    def _rescale_theta(self, tensors):
        theta, X = tensors
        shape = tf.to_float(tf.shape(X)[0:3])
        shift = 0.5
        ones = tf.constant([1. for i in range(3)])
        scale_factor = tf.concat([ones, shape / (shape + 2.)], -1)
        corrected_theta = (theta - shift) * scale_factor + shift
        return corrected_theta

    def call(self, tensors, mask=None):
        X, Y, m1, m2, theta = tensors

        M1_t = self._mask_batch_affine_warp3d(masks=m1, theta=theta)
        M2_t = self._mask_batch_affine_warp3d(masks=m2, theta=theta)

        if self.padding_method == "fill":
            # Altered to use fill method
            paddings = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
            rescale_theta = K.map_fn(self._rescale_theta, (theta, X),
                                     dtype=tf.float32)
            X = tf.pad(X, paddings, "CONSTANT")
            X_t = self._batch_affine_warp3d(imgs=X, theta=rescale_theta)
            X_t = X_t[:, 1:-1, 1:-1, 1:-1, :]

        elif self.padding_method == "replicate":
            X_t = self._batch_affine_warp3d(imgs=X, theta=theta)

        else:
            raise NotImplementedError

        output = tf.cast(self._ift3d(
            tf.math.multiply(self._ft3d(X_t), tf.cast(M1_t, tf.complex64)) +
            tf.math.multiply(self._ft3d(Y), tf.cast(M2_t, tf.complex64))),
            tf.float32)

        return [output, M1_t, M2_t]

    def get_config(self):
        config = {'output_size': self.output_size}
        base_config = super(RigidTransformation3DImputation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _ft3d(self, x):
        x_perm = tf.transpose(x, perm=[0, 4, 1, 2, 3])
        output = self._fftshift(
            self._fftshift(
                self._fftshift(
                    tf.transpose(fft(
                        tf.transpose(fft(
                            tf.transpose(fft(
                                tf.cast(x_perm, tf.complex64)),
                                perm=[0, 1, 2, 4, 3])),
                            perm=[0, 1, 3, 4, 2])),
                        perm=[0, 4, 3, 2, 1]), 1), 2), 3)

        return output

    def _ift3d(self, x):
        x_perm = tf.transpose(x, perm=[0, 4, 3, 2, 1])
        output = tf.transpose(ifft(
            tf.transpose(ifft(
                tf.transpose(ifft(
                    self._ifftshift(
                        self._ifftshift(
                            self._ifftshift(x_perm, 2), 3), 4)),
                    perm=[0, 1, 4, 2, 3])),
                perm=[0, 1, 2, 4, 3])),
            perm=[0, 2, 3, 4, 1])

        return output

    def _fftshift(self, x, axes=None):
        """
        Shift the zero-frequency component to the center of the spectrum.
        This function swaps half-spaces for all axes listed (defaults to all).
        Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

        Parameters
        ----------
        x : array_like, Tensor
            Input array.
        axes : int or shape tuple, optional
            Axes over which to shift.  Default is None, which shifts all axes.

        Returns
        -------
        y : Tensor.
        """
        if axes is None:
            axes = tuple(range(ndim(x)))
            shift = [dim // 2 for dim in tf.shape(x)]
        elif isinstance(axes, int):
            shift = tf.shape(x)[axes] // 2
        else:
            shift = [tf.shape(x)[ax] // 2 for ax in axes]

        return _roll(x, shift, axes)

    def _ifftshift(self, x, axes=None):
        """
        The inverse of `fftshift`. Although identical for even-length `x`, the
        functions differ by one sample for odd-length `x`.

        Parameters
        ----------
        x : array_like, Tensor.
        axes : int or shape tuple, optional
            Axes over which to calculate.  Defaults to None, which shifts all axes.

        Returns
        -------
        y : Tensor.
        """
        if axes is None:
            axes = tuple(range(ndim(x)))
            shift = [-(dim // 2) for dim in tf.shape(x)]
        elif isinstance(axes, int):
            shift = -(tf.shape(x)[axes] // 2)
        else:
            shift = [-(tf.shape(x)[ax] // 2) for ax in axes]

        return _roll(x, shift, axes)

    def _rotation_matrix_zyz(self, params):
        phi = params[0] * 2 * np.pi - np.pi
        theta = params[1] * 2 * np.pi - np.pi
        psi_t = params[2] * 2 * np.pi - np.pi

        loc_r = params[3:6] * 2 - 1

        # first rotate about z axis for angle psi_t
        a1 = self._rotation_matrix_axis(2, psi_t)
        a2 = self._rotation_matrix_axis(1, theta)
        a3 = self._rotation_matrix_axis(2, phi)
        rm = K.dot(K.dot(a3, a2), a1)

        rm = tf.transpose(rm)

        c = K.dot(-rm, K.expand_dims(loc_r))

        rm = K.flatten(rm)

        theta = K.concatenate([rm[:3], c[0], rm[3:6], c[1], rm[6:9], c[2]])

        return theta

    def _mask_rotation_matrix_zyz(self, params):
        phi = params[0] * 2 * np.pi - np.pi
        theta = params[1] * 2 * np.pi - np.pi
        psi_t = params[2] * 2 * np.pi - np.pi

        # magnitude of Fourier transformation is translation-invariant
        loc_r = params[3:6] * 0

        a1 = self._rotation_matrix_axis(2, psi_t)
        a2 = self._rotation_matrix_axis(1, theta)
        a3 = self._rotation_matrix_axis(2, phi)
        rm = K.dot(K.dot(a3, a2), a1)

        rm = tf.transpose(rm)

        c = K.dot(-rm, K.expand_dims(loc_r))

        rm = K.flatten(rm)

        theta = K.concatenate([rm[:3], c[0], rm[3:6], c[1], rm[6:9], c[2]])

        return theta

    def _rotation_matrix_axis(self, dim, theta):
        # following are left handed system (clockwise rotation)
        # IMPORTANT: different to MATLAB version, this dim starts from 0, instead of 1
        # x-axis
        if dim == 0:
            rm = tf.stack([[1.0, 0.0, 0.0],
                           [0.0, K.cos(theta), -K.sin(theta)],
                           [0.0, K.sin(theta), K.cos(theta)]])
        # y-axis
        elif dim == 1:
            rm = tf.stack([[K.cos(theta), 0.0, K.sin(theta)],
                           [0.0, 1.0, 0.0],
                           [-K.sin(theta), 0.0, K.cos(theta)]])
        # z-axis
        elif dim == 2:
            rm = tf.stack([[K.cos(theta), -K.sin(theta), 0.0],
                           [K.sin(theta), K.cos(theta), 0.0],
                           [0.0, 0.0, 1.0]])
        else:
            raise

        return rm

    def _interpolate3d(self, imgs, x, y, z):
        n_batch = tf.shape(imgs)[0]
        xlen = tf.shape(imgs)[1]
        ylen = tf.shape(imgs)[2]
        zlen = tf.shape(imgs)[3]
        n_channel = tf.shape(imgs)[4]

        x = tf.to_float(x)
        y = tf.to_float(y)
        z = tf.to_float(z)
        xlen_f = tf.to_float(xlen)
        ylen_f = tf.to_float(ylen)
        zlen_f = tf.to_float(zlen)
        zero = tf.zeros([], dtype='int32')
        max_x = tf.cast(xlen - 1, 'int32')
        max_y = tf.cast(ylen - 1, 'int32')
        max_z = tf.cast(zlen - 1, 'int32')

        # scale indices from [-1, 1] to [0, xlen/ylen]
        x = (x + 1.) * (xlen_f - 1.) * 0.5
        y = (y + 1.) * (ylen_f - 1.) * 0.5
        z = (z + 1.) * (zlen_f - 1.) * 0.5

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), 'int32')
        z1 = z0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        z0 = tf.clip_by_value(z0, zero, max_z)
        z1 = tf.clip_by_value(z1, zero, max_z)

        base = self._repeat(tf.range(n_batch) * xlen * ylen * zlen, xlen * ylen * zlen)
        base_x0 = base + x0 * ylen * zlen
        base_x1 = base + x1 * ylen * zlen
        base00 = base_x0 + y0 * zlen
        base01 = base_x0 + y1 * zlen
        base10 = base_x1 + y0 * zlen
        base11 = base_x1 + y1 * zlen
        index000 = base00 + z0
        index001 = base00 + z1
        index010 = base01 + z0
        index011 = base01 + z1
        index100 = base10 + z0
        index101 = base10 + z1
        index110 = base11 + z0
        index111 = base11 + z1

        # use indices to lookup pixels in the flat image and restore
        # n_channel dim
        imgs_flat = tf.reshape(imgs, [-1, n_channel])
        imgs_flat = tf.to_float(imgs_flat)
        I000 = tf.gather(imgs_flat, index000)
        I001 = tf.gather(imgs_flat, index001)
        I010 = tf.gather(imgs_flat, index010)
        I011 = tf.gather(imgs_flat, index011)
        I100 = tf.gather(imgs_flat, index100)
        I101 = tf.gather(imgs_flat, index101)
        I110 = tf.gather(imgs_flat, index110)
        I111 = tf.gather(imgs_flat, index111)

        # and finally calculate interpolated values
        dx = x - tf.to_float(x0)
        dy = y - tf.to_float(y0)
        dz = z - tf.to_float(z0)
        w000 = tf.expand_dims((1. - dx) * (1. - dy) * (1. - dz), 1)
        w001 = tf.expand_dims((1. - dx) * (1. - dy) * dz, 1)
        w010 = tf.expand_dims((1. - dx) * dy * (1. - dz), 1)
        w011 = tf.expand_dims((1. - dx) * dy * dz, 1)
        w100 = tf.expand_dims(dx * (1. - dy) * (1. - dz), 1)
        w101 = tf.expand_dims(dx * (1. - dy) * dz, 1)
        w110 = tf.expand_dims(dx * dy * (1. - dz), 1)
        w111 = tf.expand_dims(dx * dy * dz, 1)
        output = tf.add_n([
            w000 * I000, w001 * I001, w010 * I010, w011 * I011,
            w100 * I100, w101 * I101, w110 * I110, w111 * I111
        ])

        # reshape
        output = tf.reshape(output, [n_batch, xlen, ylen, zlen, n_channel])

        return output

    def _batch_warp3d(self, imgs, mappings):
        """
        warp image using mapping function
        I(x) -> I(phi(x))
        phi: mapping function
        Parameters
        ----------
        imgs : tf.Tensor
            images to be warped
            [n_batch, xlen, ylen, zlen, n_channel]
        mapping : tf.Tensor
            grids representing mapping function
            [n_batch, xlen, ylen, zlen, 3]
        Returns
        -------
        output : tf.Tensor
            warped images
            [n_batch, xlen, ylen, zlen, n_channel]
        """
        n_batch = tf.shape(imgs)[0]
        coords = tf.reshape(mappings, [n_batch, 3, -1])
        x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
        y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
        z_coords = tf.slice(coords, [0, 2, 0], [-1, 1, -1])
        x_coords_flat = tf.reshape(x_coords, [-1])
        y_coords_flat = tf.reshape(y_coords, [-1])
        z_coords_flat = tf.reshape(z_coords, [-1])

        output = self._interpolate3d(imgs, x_coords_flat,
                                     y_coords_flat, z_coords_flat)

        return output

    def _repeat(self, base_indices, n_repeats):
        base_indices = tf.matmul(tf.reshape(base_indices, [-1, 1]),
                                 tf.ones([1, n_repeats], dtype='int32'))

        return tf.reshape(base_indices, [-1])

    def _mgrid(self, *args, **kwargs):
        """
        create orthogonal grid
        similar to np.mgrid

        Parameters
        ----------
        args : int
            number of points on each axis
        low : float
            minimum coordinate value
        high : float
            maximum coordinate value

        Returns
        -------
        grid : tf.Tensor [len(args), args[0], ...]
            orthogonal grid
        """
        low = kwargs.pop("low", -1)
        high = kwargs.pop("high", 1)
        low = tf.to_float(low)
        high = tf.to_float(high)
        coords = (tf.linspace(low, high, arg) for arg in args)
        grid = tf.stack(tf.meshgrid(*coords, indexing='ij'))

        return grid

    def _batch_mgrid(self, n_batch, *args, **kwargs):
        """
        create batch of orthogonal grids
        similar to np.mgrid

        Parameters
        ----------
        n_batch : int
            number of grids to create
        args : int
            number of points on each axis
        low : float
            minimum coordinate value
        high : float
            maximum coordinate value

        Returns
        -------
        grids : tf.Tensor [n_batch, len(args), args[0], ...]
            batch of orthogonal grids
        """
        grid = self._mgrid(*args, **kwargs)
        grid = tf.expand_dims(grid, 0)
        grids = tf.tile(grid, [n_batch] + [1 for _ in range(len(args) + 1)])

        return grids

    def _batch_affine_warp3d(self, imgs, theta):
        """
        affine transforms 3d images

        Parameters
        ----------
        imgs : tf.Tensor
            images to be warped
            [n_batch, xlen, ylen, zlen, n_channel]
        theta : tf.Tensor
            parameters of affine transformation
            [n_batch, 12]

        Returns
        -------
        output : tf.Tensor
            warped images
            [n_batch, xlen, ylen, zlen, n_channel]
        """
        n_batch = tf.shape(imgs)[0]
        xlen = tf.shape(imgs)[1]
        ylen = tf.shape(imgs)[2]
        zlen = tf.shape(imgs)[3]

        c = K.map_fn(self._rotation_matrix_zyz, theta)

        theta = tf.reshape(c, [-1, 3, 4])
        matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 3])
        t = tf.slice(theta, [0, 0, 3], [-1, -1, -1])

        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen)
        grids = tf.reshape(grids, [n_batch, 3, -1])

        T_g = tf.matmul(matrix, grids) + t
        T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
        output = self._batch_warp3d(imgs, T_g)

        return output

    def _mask_batch_affine_warp3d(self, masks, theta):
        """
        affine transforms 3d images

        Parameters
        ----------
        imgs : tf.Tensor
            images to be warped
            [n_batch, xlen, ylen, zlen, n_channel]
        theta : tf.Tensor
            parameters of affine transformation
            [n_batch, 12]

        Returns
        -------
        output : tf.Tensor
            warped images
            [n_batch, xlen, ylen, zlen, n_channel]
        """
        n_batch = tf.shape(masks)[0]
        xlen = tf.shape(masks)[1]
        ylen = tf.shape(masks)[2]
        zlen = tf.shape(masks)[3]

        c = K.map_fn(self._mask_rotation_matrix_zyz, theta)

        theta = tf.reshape(c, [-1, 3, 4])
        matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 3])
        t = tf.slice(theta, [0, 0, 3], [-1, -1, -1])

        grids = self._batch_mgrid(n_batch, xlen, ylen, zlen)
        grids = tf.reshape(grids, [n_batch, 3, -1])

        T_g = tf.matmul(matrix, grids) + t
        T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
        output = self._batch_warp3d(masks, T_g)

        return output
