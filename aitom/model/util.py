"""
Utility function for generate different geometirc models
"""

import numpy as N
import aitom.image.vol.util as gv
import aitom.geometry.rotate as GR


def gauss_function(size, sigma):
    grid = gv.grid_displacement_to_center(size)
    dist_sq = gv.grid_distance_sq_to_center(grid)

    del grid

    # gauss function
    g = (1 / ((2 * N.pi) ** (3.0 / 2.0) * (sigma ** 3))) * N.exp(- (dist_sq) / (2.0 * (sigma ** 2)))

    return g


def generate_toy_model(dim_siz=64, model_id=0):
    """translated from SphericalHarmonicsUtil.generate_toy_vol()"""
    siz = N.array([dim_siz, dim_siz, dim_siz])

    mid = siz / 2.0

    xg = N.mgrid[0:siz[0], 0:siz[1], 0:siz[2]]

    if model_id == 0:
        # four gauss functions
        short_dia = 0.4
        mid_dia = 0.8
        long_dia = 1.2

        e0 = generate_toy_model__gaussian(dim_siz=dim_siz, xg=xg, xm=(mid + N.array([siz[0] / 4.0, 0.0, 0.0])),
                                          dias=[long_dia, short_dia, short_dia])
        e1 = generate_toy_model__gaussian(dim_siz=dim_siz, xg=xg, xm=(mid + N.array([0.0, siz[1] / 4.0, 0.0])),
                                          dias=[short_dia, long_dia, short_dia])
        e2 = generate_toy_model__gaussian(dim_siz=dim_siz, xg=xg, xm=(mid + N.array([0.0, 0.0, siz[2] / 4.0])),
                                          dias=[short_dia, short_dia, long_dia])

        e3 = GR.rotate_pad_zero(N.array(e0, order='F'), angle=N.array([N.pi / 4.0, 0.0, 0.0]),
                                loc_r=N.array([0.0, 0.0, 0.0]))

        e = e0 + e1 + e2 + e3

    return e


def generate_toy_model__gaussian(dim_siz, xg, xm, dias):
    x = N.zeros(xg.shape)
    for dim_i in range(3):
        x[dim_i] = xg[dim_i] - xm[dim_i]
    xs = N.array([x[0] / (dim_siz * dias[0]), x[1] / (dim_siz * dias[1]), x[2] / (dim_siz * dias[2])])
    e = N.exp(- N.sum(xs * x, axis=0))

    return e


def sphere_mask(shape, center=None, radius=None, smooth_sigma=None):
    shape = N.array(shape)

    v = N.zeros(shape)

    if center is None:
        # IMPORTANT: following python convension, in index starts from 0 to
        # size-1 !!! So (siz-1)/2 is real symmetry center of the volume
        center = (shape - 1) / 2.0

    center = N.array(center)

    if radius is None:
        radius = N.min(shape / 2.0)

    grid = gv.grid_displacement_to_center(shape, mid_co=center)
    dist = gv.grid_distance_to_center(grid)

    v[dist <= radius] = 1.0

    if smooth_sigma is not None:

        assert smooth_sigma > 0
        v_s = N.exp(-((dist - radius) / smooth_sigma) ** 2)
        # use a cutoff of -3 looks nicer, although the tom toolbox uses -2
        v_s[v_s < N.exp(-3)] = 0.0
        v[dist >= radius] = v_s[dist >= radius]

    return v
