"""
construct a missing wedge mask, see tom_wedge,
angle represents the angle range of MISSING WEDGE region, the larger, the more missing wedge region!!!
tilt_axis is tilt axis
"""

import warnings
import numpy as N
from . import util as AIVU
import aitom.model.util as MU


def wedge_mask(size, ang1, ang2=None, tilt_axis=1, sphere_mask=True, verbose=False):
    # should define both tilt axis and electron beam (missing wedge) direction
    warnings.warn("The definition of wedge mask is still ambiguous")

    if ang2 is None:
        ang2 = float(N.abs(ang1))
        ang1 = -ang2

    else:
        assert ang1 < 0
        assert ang2 > 0

    if verbose:
        print('image.vol.wedge.util.wedge_mask()', 'ang1', ang1, 'ang2', ang2, 'tilt_axis', tilt_axis, 'sphere_mask',
              sphere_mask)

    ang1 = (ang1 / 180.0) * N.pi
    ang2 = (ang2 / 180.0) * N.pi

    g = AIVU.grid_displacement_to_center(size=size, mid_co=AIVU.fft_mid_co(siz=size))

    if tilt_axis == 0:
        # y-z plane
        # y axis
        x0 = g[1]
        # z axis
        x1 = g[2]

    elif tilt_axis == 1:
        # x-z plane
        # x axis
        x0 = g[0]
        # z axis
        x1 = g[2]

    elif tilt_axis == 2:
        # x-y plane
        # x axis
        x0 = g[0]
        # y axis
        x1 = g[1]

    m = N.zeros(size, dtype=float)

    m[N.logical_and(x0 >= (N.tan(ang2) * x1), x0 >= (N.tan(ang1) * x1))] = 1.0
    m[N.logical_and(x0 <= (N.tan(ang1) * x1), x0 <= (N.tan(ang2) * x1))] = 1.0

    if sphere_mask:
        m *= MU.sphere_mask(m.shape)

    return m


def tilt_mask(size, tilt_ang1, tilt_ang2=None, tilt_axis=1, light_axis=2, sphere_mask=True):
    """wedge mask defined using tilt angles light axis is the direction of electrons"""
    assert tilt_axis != light_axis

    if tilt_ang2 is None:
        tilt_ang2 = float(N.abs(tilt_ang1))
        tilt_ang1 = -tilt_ang2

    else:
        assert tilt_ang1 < 0
        assert tilt_ang2 > 0

    tilt_ang1 = (tilt_ang1 / 180.0) * N.pi
    tilt_ang2 = (tilt_ang2 / 180.0) * N.pi

    g = AIVU.grid_displacement_to_center(size=size, mid_co=IVU.fft_mid_co(siz=size))

    plane_axis = set([0, 1, 2])
    plane_axis.difference_update([light_axis, tilt_axis])
    assert len(plane_axis) == 1
    plane_axis = list(plane_axis)[0]

    x_light = g[light_axis]
    x_plane = g[plane_axis]

    m = N.zeros(size, dtype=float)

    m[N.logical_and(x_light <= (N.tan(tilt_ang1) * x_plane), x_light >= (N.tan(tilt_ang2) * x_plane))] = 1.0
    m[N.logical_and(x_light >= (N.tan(tilt_ang1) * x_plane), x_light <= (N.tan(tilt_ang2) * x_plane))] = 1.0

    if sphere_mask:
        m *= MU.sphere_mask(m.shape)

    return m
