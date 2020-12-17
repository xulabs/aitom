"""
functions for calculating different kinds of differentials in 2D and 3D images
"""

import numpy as N


def diff_one_axis(v, axis):
    s = v.shape
    vd = N.zeros(s)

    if axis == 0:
        vd[:(s[0] - 1), :, :] = N.diff(v, axis=axis)
    elif axis == 1:
        vd[:, :(s[1] - 1), :] = N.diff(v, axis=axis)
    elif axis == 2:
        vd[:, :, :(s[2] - 1)] = N.diff(v, axis=axis)
    else:
        raise

    return vd


# calculate gradient by differential
def diff_3d(v):
    d = []
    for dim in range(3):
        d.append(diff_one_axis(v, dim))

    return d


# the square of gradient magnitude, d=diff_3d(v)
def gradient_magnitude_square(d):
    s = N.zeros(d[0].shape)
    for dt in d:
        s += N.square(dt)

    return s


def gradient_normal(d):
    m = N.sqrt(gradient_magnitude_square(d))

    ind = m > 0

    dn = [None] * len(d)
    for dim in range(len(d)):
        dn[dim] = N.zeros(d[dim].shape)
        dn[dim][ind] = d[dim][ind] / m[ind]

    return dn


# see directional dirvative:  https://en.wikipedia.org/wiki/Directional_derivative
def directional_derivative_along_gradient(v, d):
    d = gradient_normal(d)
    vd = diff_3d(v)

    dd = N.zeros(v.shape)
    for dim in range(len(d)):
        dd += vd[dim] * d[dim]

    return dd


def hessian_3d(v, d=None):
    if d is None:
        d = diff_3d(v)

    h = [[None] * 3] * 3
    for dim0 in range(3):
        for dim1 in range(3):
            if dim0 > dim1:
                continue
            h[dim0][dim1] = diff_one_axis(d[dim0], dim1)

    return h


def hessian_3d__max_magnitude(h):
    m = 0
    for ht in h:
        for htt in ht:
            if htt is None:
                continue
            m = N.max((m, N.abs(htt).max()))

    return float(m)


def hessian_3d__normalize(h, magnitude):
    """
    dividing the max magnitude so that all entries have less than 1.0 magnitude, this is needed by
    linalg.eigen.eigen_value_3_symmetric_batch()
    """
    ht = [[None] * 3] * 3
    for dim0 in range(3):
        for dim1 in range(3):
            if h[dim0][dim1] is None:
                continue
            ht[dim0][dim1] = h[dim0][dim1] / magnitude

    return ht
