


import numpy as N
import scipy.ndimage.interpolation as SNI
import aitom.geometry.ang_loc as AA
import aitom.tomominer.image.vol.util as IVU

def rotate(v, angle=None, rm=None, c1=None, c2=None, loc_r=None, siz2=None, default_val=float('NaN')):
    if (angle is not None):
        assert (rm is None)
        angle = N.array(angle, dtype=N.float).flatten()
        rm = AA.rotation_matrix_zyz(angle)
    if (rm is None):
        rm = N.eye(v.ndim)
    siz1 = N.array(v.shape, dtype=N.float)
    if (c1 is None):
        c1 = ((siz1 - 1) / 2.0)
    else:
        c1 = c1.flatten()
    assert (c1.shape == (3,))
    if (siz2 is None):
        siz2 = siz1
    siz2 = N.array(siz2, dtype=N.float)
    if (c2 is None):
        c2 = ((siz2 - 1) / 2.0)
    else:
        c2 = c2.flatten()
    assert (c2.shape == (3,))
    if (loc_r is not None):
        loc_r = N.array(loc_r, dtype=N.float).flatten()
        assert (loc_r.shape == (3,))
        c2 += loc_r
    c = ((- rm.dot(c2)) + c1)
    vr = SNI.affine_transform(input=v, matrix=rm, offset=c, output_shape=siz2.astype(N.int), cval=default_val)
    return vr

def rotate_pad_mean(v, angle=None, rm=None, c1=None, c2=None, loc_r=None, siz2=None):
    vr = rotate(v, angle=angle, rm=rm, c1=c1, c2=c2, loc_r=loc_r, siz2=siz2, default_val=float('NaN'))
    if False:
        vr[N.logical_not(N.isfinite(vr))] = vr[N.isfinite(vr)].mean()
    else:
        vr[N.logical_not(N.isfinite(vr))] = v.mean()
    return vr

def rotate_pad_zero(v, angle=None, rm=None, c1=None, c2=None, loc_r=None, siz2=None):
    vr = rotate(v, angle=angle, rm=rm, c1=c1, c2=c2, loc_r=loc_r, siz2=siz2, default_val=float('NaN'))

    vr[N.logical_not(N.isfinite(vr))] = 0.0

    return vr


def rotate_mask(v, angle=None, rm=None):
    c1 = IVU.fft_mid_co(v.shape)
    c2 = N.copy(c1)
    vr = rotate(v, angle=angle, rm=rm, c1=c1, c2=c2, default_val=float('NaN'))
    vr[N.logical_not(N.isfinite(vr))] = 0.0
    return vr
