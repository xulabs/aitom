'''
construct a missing wedge mask, see tom_wedge,
angle represents the angle range of MISSING WEDGE region, the larger, the more missing wedge region!!!
tilt_axis is tilt axis
'''
import warnings
import numpy as N
import aitom.image.vol.util as AIVU
import aitom.model.util as MU

def wedge_mask(size, ang1, ang2=None, tilt_axis=1, sphere_mask=True, verbose=False):
    warnings.warn("The definition of wedge mask is still ambiguous")        # should define both tilt axis and electron beam (missing wedge) direction

    if ang2 is None:
        ang2 = float(N.abs(ang1))
        ang1 = -ang2

    else:
        assert      ang1 < 0
        assert      ang2 > 0


    if verbose:     print('image.vol.wedge.util.wedge_mask()', 'ang1', ang1, 'ang2', ang2, 'tilt_axis', tilt_axis, 'sphere_mask', sphere_mask)

    ang1 = (ang1 / 180.0) * N.pi
    ang2 = (ang2 / 180.0) * N.pi

    g = AIVU.grid_displacement_to_center(size=size, mid_co=AIVU.fft_mid_co(siz=size))

    if tilt_axis==0:
        # y-z plane
        x0 = g[1]           # y axis
        x1 = g[2]           # z axis

    elif tilt_axis==1:
        # x-z plane
        x0 = g[0]       # x axis
        x1 = g[2]       # z axis

    elif tilt_axis==2:
        # x-y plane
        x0 = g[0]       # x axis
        x1 = g[1]       # y axis

    m = N.zeros(size, dtype=float)

    m[ N.logical_and(x0 >= (N.tan(ang2)*x1), x0 >= (N.tan(ang1)*x1)) ] = 1.0
    m[ N.logical_and(x0 <= (N.tan(ang1)*x1), x0 <= (N.tan(ang2)*x1)) ] = 1.0

    if sphere_mask:    m *= MU.sphere_mask(m.shape)

    return m


