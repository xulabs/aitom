import numpy as N
import scipy.ndimage.interpolation as SNI
from . import ang_loc as AA
from ..image.vol import util as IVU


def rotate(v, angle=None, rm=None, c1=None, c2=None, loc_r=None, siz2=None, default_val=float('NaN')):
    if angle is not None:
        assert (rm is None)
        angle = N.array(angle, dtype=N.float).flatten()
        rm = AA.rotation_matrix_zyz(angle)
    if rm is None:
        rm = N.eye(v.ndim)
    siz1 = N.array(v.shape, dtype=N.float)
    if c1 is None:
        c1 = ((siz1 - 1) / 2.0)
    else:
        c1 = c1.flatten()
    assert (c1.shape == (3,))
    if siz2 is None:
        siz2 = siz1
    siz2 = N.array(siz2, dtype=N.float)
    if c2 is None:
        c2 = ((siz2 - 1) / 2.0)
    else:
        c2 = c2.flatten()
    assert (c2.shape == (3,))
    if loc_r is not None:
        loc_r = N.array(loc_r, dtype=N.float).flatten()
        assert (loc_r.shape == (3,))
        c2 += loc_r
    c = ((- rm.dot(c2)) + c1)
    vr = SNI.affine_transform(input=v, matrix=rm, offset=c, output_shape=siz2.astype(N.int), cval=default_val)
    return vr


def rotate3d_zyz(data, angle=None, rm=None, center=None, order=2, cval=0.0):
    """Rotate a 3D data using ZYZ convention (phi: z1, the: x, psi: z2)."""
    from scipy import mgrid
    # Figure out the rotation center
    if center is None:
        cx = data.shape[0] / 2
        cy = data.shape[1] / 2
        cz = data.shape[2] / 2
    else:
        assert len(center) == 3
        (cx, cy, cz) = center

    if rm is None:
        Inv_R = AA.rotation_matrix_zyz(angle)
    else:
        Inv_R = rm

    grid = mgrid[-cx:data.shape[0] - cx, -cy:data.shape[1] - cy, -cz:data.shape[2] - cz]
    temp = grid.reshape((3, N.int(grid.size / 3)))
    temp = N.dot(Inv_R, temp)
    grid = N.reshape(temp, grid.shape)
    grid[0] += cx
    grid[1] += cy
    grid[2] += cz

    # Interpolation
    from scipy.ndimage import map_coordinates
    d = map_coordinates(data, grid, order=order, cval=cval)

    return d


def translate3d_zyz(data, dx=0, dy=0, dz=0, order=2, cval=0.0):
    """Translate the data.
    @param
        data: data to be shifted.
        dx: translation along x-axis.
        dy: translation along y-axis.
        dz: translation along z-axis.
    
    @return: the data after translation.
    """
    from scipy import mgrid
    if dx == 0 and dy == 0 and dz == 0:
        return data

    # from scipy.ndimage.interpolation import shift
    # res = shift(data, [dx, dy, dz])
    # return res
    grid = mgrid[0.:data.shape[0], 0.:data.shape[1], 0.:data.shape[2]]
    grid[0] -= dx
    grid[1] -= dy
    grid[2] -= dz
    from scipy.ndimage import map_coordinates
    d = map_coordinates(data, grid, order=order, cval=cval)

    return d


def rotate_interpolate_pad_mean(v, angle=None, rm=None, loc_r=None):
    cval = v.mean()

    vr = rotate3d_zyz(v, angle=angle, cval=cval)

    vr = translate3d_zyz(vr, loc_r[0], loc_r[1], loc_r[2], cval=cval)

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
