
import numpy as N
import numpy.fft as NF


import aitom.model.util as TMU


'''
adapted according to 
/opt/local/img/em/et/util/matlab_tom/2008/TOM_Release_2008/Filtrans/tom_bandpass.m
'''
import aitom.image.vol.util as TIVU
def tom_bandpass(v, low, hi, smooth=None):

    vt = NF.fftn(v)
    vt = NF.fftshift(vt)

    mid_co = TIVU.fft_mid_co(v.shape)
    if smooth is None:
        d = TIVU.grid_distance_sq_to_center(TIVU.grid_displacement_to_center(v.shape, mid_co=mid_co))
        vt[d > hi] = 0.0
        vt[d < low] = 0.0
    else:
        m = TMU.sphere_mask(v.shape, center=mid_co, radius=hi, smooth_sigma=smooth)
        if low > 0:            m -= TMU.sphere_mask(v.shape, center=mid_co, radius=low, smooth_sigma=smooth)
        vt *= m

    vt = NF.ifftshift(vt)
    vt = NF.ifftn(vt)
    vt = N.real(vt)
    return vt


