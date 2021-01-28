"""
2D and 3D ctf function, written according to
~/ln/electron/matlab_tom/2005/Analysis/tom_create_ctf.m
~/ln/tomominer/tomominer/image/optics/ctf.py
"""

import numpy as N
from numpy.fft import fftn, ifftn, fftshift, ifftshift


def create(Dz, size, pix_size=0.72, voltage=300.0, Cs=2.0, sigma=None, display_info=False):
    """
    Param:
        Dz       : Defocus (<0 underfocus, >0 overfocus) (in \mu m);
        vol      : Volume (or image)
        pix_size : pixel size (in nm) (default: 0.72 nm)
        voltage  : accelerating Voltage (in keV) (default: 300 kV)
        Cs       : sperical aberration (in mm)
        Sigma    : envelope of ctf (optional). If chosen decay
                   ctf ~exp(-(freq/sigma)^2) is assumed. SIGMA is in units of Nyqvist.
                   => ctf(sigma) = 1/e

    Result:
        ctf_out  : output containing the centrosymmetric ctf. It can be used asa fourier filter.
    """

    Cs *= 1e-3
    voltage *= 1000.0
    pix_size *= 1e-9

    Dz *= 1e-6
    Dzn = Dz * 1000000.0  # for display
    Csn = Cs * 1000.0  # for display
    voltagen = voltage / 1000.0  # for display
    voltagest = voltage * (1.0 + voltage / 1022000.0)  # for relativistic calc
    lambda_t = N.sqrt(150.4 / voltagest) * 1e-10

    Ny = 1.0 / (2.0 * pix_size)
    nyqvist = 2.0 * pix_size * 1e9

    if display_info:
        print('CTF is calculated for: Defocus', Dzn, '\mum Voltage = ', voltagen, 'kV, Nyqvist = ', nyqvist, 'nm')

    if len(size) == 2:
        g = N.mgrid[0:size[0], 0:size[1]]
    elif len(size) == 3:
        g = N.mgrid[0:size[0], 0:size[1], 0:size[2]]
    else:
        raise Exception('2D or 3D array only')

    g = g.astype(N.float)

    for dim_i in range(len(g)):
        g[dim_i] *= Ny / (size[dim_i] / 2.0)
        g[dim_i] -= Ny

    r = N.sqrt((g ** 2).sum(axis=0))

    vol = N.sin((N.pi / 2.0) * (Cs * (lambda_t ** 3.0) * (r ** 4.0) - 2.0 * Dz * lambda_t * (r ** 2.0)))
    amplitude = N.cos((N.pi / 2.0) * (Cs * (lambda_t ** 3.0) * (r ** 4.0) - 2.0 * Dz * lambda_t * (r ** 2.0)))

    if sigma is not None:
        # gaussian smoothing....
        vol *= N.exp(-(r / (sigma * Ny)) ** 2.0)
        amplitude *= N.exp(-(r / (sigma * Ny)) ** 2.0)

    assert N.all(N.isfinite(vol))

    return {'ctf': vol, 'amplitude': amplitude, 'g': g}


def apply_ctf(v, ctf):
    """apply CTF to a given density map"""
    vc = N.real(ifftn(ifftshift(ctf * fftshift(fftn(v)))))  # convolute v with ctf
    return vc
