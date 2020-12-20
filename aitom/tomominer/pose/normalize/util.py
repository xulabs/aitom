"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import numpy as N
import aitom.tomominer.rotate as GR


def center_mass(v):
    assert N.all((v >= 0))
    m = v.sum()
    assert (m > 0)
    v = (v / m)
    s = v.shape
    g = N.mgrid[0:s[0], 0:s[1], 0:s[2]]
    c = ([None] * v.ndim)
    for dim_i in range(v.ndim):
        c[dim_i] = (g[dim_i] * v).sum()
    return N.array(c)


def pca(v, c, do_flip=False):
    assert N.all((v >= 0))
    re = {'c': c, }
    s = v.shape
    g = N.mgrid[0:s[0], 0:s[1], 0:s[2]]
    g = N.array(g, dtype=N.float)
    for i in range(len(g)):
        g[i] -= c[i]
    gv = []
    gv = [g[0].flatten(), g[1].flatten(), g[2].flatten()]
    vv = v.flatten()
    gvw = [(_ * vv) for _ in gv]
    wsm = N.dot(N.array(gv), N.array(gvw).T)
    re['wsm'] = wsm
    (eig_w, eig_v) = N.linalg.eig(wsm)
    i = N.argsort((- N.abs(eig_w)))
    eig_w = eig_w[i]
    eig_v = eig_v[:, i]
    re['w'] = eig_w
    if do_flip:
        re['v'] = flip_sign(v=v, c=c, r=eig_v)
    else:
        re['v'] = eig_v
    return re
