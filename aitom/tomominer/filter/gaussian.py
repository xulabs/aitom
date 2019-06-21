

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import gc as GC
import numpy as N
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import tomominer.image.vol.util as CV
import tomominer.model.util as MU
import scipy.ndimage as SN
import scipy.ndimage.filters as SNF

def smooth(v, sigma):
    assert (sigma > 0)
    return SN.gaussian_filter(input=v, sigma=sigma)