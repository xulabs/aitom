"""
statistics of volumes
"""

import numpy as N
import numpy.fft as NF
import aitom.image.vol.util as GV


def fsc(v1, v2, band_width_radius=1.0):
    """Fourier Shell correlation between two volumes"""
    siz = v1.shape
    assert (siz == v2.shape)

    origin_co = GV.fft_mid_co(siz)

    x = N.mgrid[0:siz[0], 0:siz[1], 0:siz[2]]
    x = x.astype(N.float)

    for dim_i in range(3):
        x[dim_i] -= origin_co[dim_i]

    rad = N.sqrt(N.square(x).sum(axis=0))

    vol_rad = int(N.floor(N.min(siz) / 2.0) + 1)

    v1f = NF.fftshift(NF.fftn(v1))
    v2f = NF.fftshift(NF.fftn(v2))

    fsc_cors = N.zeros(vol_rad)

    # the interpolation can also be performed using scipy.ndimage.interpolation.map_coordinates()
    for r in range(vol_rad):

        ind = (abs(rad - r) <= band_width_radius)

        c1 = v1f[ind]
        c2 = v2f[ind]

        fsc_cor_t = N.sum(c1 * N.conj(c2)) / N.sqrt(N.sum(N.abs(c1) ** 2) * N.sum(N.abs(c2) ** 2))
        fsc_cors[r] = N.real(fsc_cor_t)

    return fsc_cors
