"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import os
import pickle
import time
import random
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import aitom.tomominer.io.file as iv
import aitom.tomominer.average.util as avgu
import aitom.image.vol.util as uv


def neighbor_product(v):
    siz = list(v.shape)
    siz.append(26)
    p = np.zeros(siz, dtype=np.float32)
    shift = np.zeros((26, 3), dtype=np.int8)
    i = 0
    for s0 in range((-1), 2):
        for s1 in range((-1), 2):
            for s2 in range((-1), 2):
                if (s0 == 0) and (s1 == 0) and (s2 == 0):
                    continue
                p[:, :, :, i] = (v * uv.roll(v, s0, s1, s2))
                shift[i, :] = np.array([s0, s1, s2])
                i += 1
    assert (i == 26)
    return {'p': p, 'shift': shift, }
