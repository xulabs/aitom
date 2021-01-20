"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import sys
import numpy as N
import aitom.tomominer.core.core as core


def segment(I, phi, smooth_weight, image_weight, delta_t=1.0, n_iters=20, mean_values=None, print_progress=False):
    if not I.flags.f_contiguous:
        I = I.copy(order='F')
    assert N.all(N.isfinite(I))
    assert N.all(N.isfinite(phi))
    phi = N.array(phi, dtype=N.float, order='F')
    for i in range(n_iters):
        phi_old = phi
        if mean_values is None:
            if (phi >= 0).sum() == 0:
                return
            mu_in = N.mean(I[(phi >= 0)])
            if not N.isfinite(mu_in):
                return
            if (phi < 0).sum() == 0:
                return
            mu_out = N.mean(I[(phi < 0)])
            if not N.isfinite(mu_out):
                return
        else:
            mu_out = mean_values[0]
            mu_in = mean_values[1]
        phi = ac_reinit(phi)
        phi = (phi + ((delta_t * image_weight) * (((I - mu_out) ** 2) - ((I - mu_in) ** 2))))
        phi = ac_reinit(phi)
        phi = ac_laplacian_AOS(phi, (delta_t * smooth_weight), 1)
        assert N.all(N.isfinite(phi))
        if N.all(((phi > 0) == (phi_old > 0))):
            break
        if print_progress:
            sys.stdout.write(('\r%d ' % i))
            sys.stdout.flush()
    phi[0, :, :] = phi[1, :, :]
    phi[(phi.shape[0] - 1), :, :] = phi[(phi.shape[0] - 2), :, :]
    phi[:, 0, :] = phi[:, 1, :]
    phi[:, (phi.shape[1] - 1), :] = phi[:, (phi.shape[1] - 2), :]
    phi[:, :, 0] = phi[:, :, 1]
    phi[:, :, (phi.shape[2] - 1)] = phi[:, :, (phi.shape[2] - 2)]
    return phi


def ac_reinit(u):
    m = N.array((u > 0), dtype=N.uint8)
    u0 = core.zy_binary_boundary_detection(m)
    u = (core.ac_distance_transform_3d(u0) * N.sign(u))
    return u


def ac_laplacian_AOS(u, delta_t, n_iters):
    g = N.ones(u.shape, dtype=N.float)
    for i in range(n_iters):
        u = core.ac_div_AOS_3D(u, g, delta_t)
    return u


def segment_with_postprocessing(v, op):
    if 'mean_values' not in op:
        op['mean_values'] = None
    if 'n_iters' not in op:
        op['n_iters'] = 20
    if 'print_progress' not in op:
        op['print_progress'] = False
    v_var = v.var()
    if v_var <= 0:
        return
    phi = N.sign((v - N.abs(v).mean()))
    phi = segment(v, phi, op['smooth_weight'], (op['image_weight'] / v_var), mean_values=op['mean_values'],
                  n_iters=op['n_iters'], print_progress=op['print_progress'])
    if phi is None:
        return
    if (phi > 0).sum() == 0:
        return
    if (phi < 0).sum() == 0:
        return
    if v[(phi > 0)].mean() < v[(phi < 0)].mean():
        phi = (- phi)
    return phi
