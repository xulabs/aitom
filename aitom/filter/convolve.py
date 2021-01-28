"""
functions to support convolution
"""

from ..image.vol import util as IVU
import numpy as N
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import matplotlib
from ..geometry import ang_loc as GA
from ..geometry import rotate as GR
from ..model import util as MU


def convolve(v, t):
    """convolution, v is the map, t is the template to convolve"""
    if t.shape != v.shape:
        tf = N.zeros(v.shape)
        se = IVU.paste_to_whole_map(whole_map=tf, vol=t)
    else:
        tf = t

    tf_fft = fftn(ifftshift(tf))
    del tf

    v_conv = N.real(ifftn(fftn(v) * N.conj(tf_fft)))
    del tf_fft

    return v_conv


# # test convolve() and scipy.ndimage.filters.convolve()
# enlarge_factor = 2
# t = MU.generate_toy_model()
#
# tr = GR.rotate(t, rm=GA.random_rotation_matrix(), default_val=0.0)
# v = GR.rotate(tr, loc_r=(N.random.rand(3) - 0.5) * N.array(t.shape) * enlarge_factor / 2,
#               siz2=N.array(t.shape) * enlarge_factor, default_val=0.0)
# v += GR.rotate(tr, loc_r=(N.random.rand(3) - 0.5) * N.array(t.shape) * enlarge_factor / 2,
#                siz2=N.array(t.shape) * enlarge_factor, default_val=0.0)
# if False:
#     import scipy.ndimage.filters as SNF
#
#     vc = SNF.convolve(input=v, weights=tr)
# else:
#     from . import convolve as IC
#
#     vc = IC.convolve(v=v, t=tr)
#
# matplotlib.use('Qt4Agg')
# IVU.dsp_cub(v)
# IVU.dsp_cub(vc)


def pearson_correlation_simple(v, t):
    c = convolve(v, t)
    c /= v.size

    return c / (v.std() * t.std())
