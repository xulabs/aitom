"""
functions for gaussian filtering
"""

import scipy.ndimage as SN
import numpy as N
import gc
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from ..model import util as MU
from ..classify.deep.unsupervised.autoencoder.autoencoder_util import difference_of_gauss_function
from ..image.vol.util import paste_to_whole_map


def smooth(v, sigma):
    """smoothing using scipy.ndimage.gaussian_filter"""
    assert sigma > 0
    return SN.gaussian_filter(input=v, sigma=sigma)


def dog_smooth(v, s1, s2=None):
    """Difference of gaussian filter"""
    if s2 is None:
        s2 = s1 * 1.1  # the 1.1 is according to a DoG particle picking paper
    assert s1 < s2
    return smooth(v, s1) - smooth(v, s2)


def dog_smooth__large_map(v, s1, s2=None):
    """
    convolve with a dog function, delete unused data when necessary
    in order to save memory for large maps
    """
    if s2 is None:
        s2 = s1 * 1.1  # the 1.1 is according to a DoG particle picking paper
    assert s1 < s2

    size = v.shape

    pad_width = int(N.round(s2 * 2))
    vp = N.pad(array=v, pad_width=pad_width, mode='reflect')

    v_fft = fftn(vp).astype(N.complex64)
    del v
    gc.collect()

    g_small = difference_of_gauss_function(size=N.array([int(N.round(s2 * 4))] * 3), sigma1=s1, sigma2=s2)
    assert N.all(N.array(g_small.shape) <= N.array(vp.shape))  # make sure we can use CV.paste_to_whole_map()

    g = N.zeros(vp.shape)
    paste_to_whole_map(whole_map=g, vol=g_small, c=None)

    g_fft_conj = N.conj(fftn(ifftshift(g)).astype(N.complex64))  # use ifftshift(g) to move center of gaussian to origin
    del g
    gc.collect()

    prod_t = (v_fft * g_fft_conj).astype(N.complex64)
    del v_fft
    gc.collect()
    del g_fft_conj
    gc.collect()

    prod_t_ifft = ifftn(prod_t).astype(N.complex64)
    del prod_t
    gc.collect()

    v_conv = N.real(prod_t_ifft)
    del prod_t_ifft
    gc.collect()
    v_conv = v_conv.astype(N.float32)

    v_conv = v_conv[(pad_width + 1):(pad_width + size[0] + 1), (pad_width + 1):(pad_width + size[1] + 1),
             (pad_width + 1):(pad_width + size[2] + 1)]
    assert size == v_conv.shape

    return v_conv
