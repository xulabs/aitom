"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import numpy as N
import warnings
import exceptions


def fit__zero_mean(x, y, a0=None, c0=None, tolerance=0.001, lambda_t=10.0, max_iter_num=1000, c_bound_max=100,
                   verbose=False):
    assert (len(x) == len(y))
    i = N.argsort(x)
    x = x[i]
    y = y[i]
    x_abs_max = N.abs(x).max()
    if a0 is None:
        a0 = y.max()
    if c0 is None:
        c0 = N.median(x)
        warnings.warn('such initial guess of c0 is not stable', exceptions.Warning)
    a = a0
    c = c0
    if verbose:
        print('a0', a0, '   ', 'c0', c0)
    iter_n = 0
    e = None
    e0 = None
    e_old = None
    while True:
        e_old = e
        exp_t = N.exp(((- N.square(x)) / ((2 * c) * c)))
        yp = (a * exp_t)
        e = N.sqrt(N.square((y - yp)).sum())
        if e0 is None:
            e0 = e
        if e_old is not None:
            e_rate = (N.abs((e - e_old)) / e0)
            if e_rate < tolerance:
                break
        ga = exp_t
        gc = (((a * exp_t) * N.square(x)) / (c ** 3))
        J = N.zeros((len(x), 2))
        J[:, 0] = ga
        J[:, 1] = gc
        Jq = N.dot(J.T, J)
        d = N.linalg.solve((Jq + (lambda_t * N.diag(Jq))), N.dot(J.T, (y - yp)))
        if verbose and (e_old is not None):
            print('\r', 'd', d, '    ', 'a', a, '    ', 'c', c, '    ', 'e', e, '    ', 'e_old', e_old, '    ', 'e0',
                  e0, '    ', 'e_rate', e_rate, '           ', end=' ')
        a += d[0]
        c += d[1]
        if N.abs(c) > (x_abs_max * c_bound_max):
            return {'a': a, 'c': N.abs(c), 'e': e, }
        iter_n += 1
        if iter_n > max_iter_num:
            break
    return {'a': a, 'c': N.abs(c), 'e': e, }


def fit__zero_mean__gaussian_function(x, a, c):
    return a * N.exp(((- N.square(x)) / ((2 * c) * c)))


def fit__zero_mean__multi_start(x, y, tolerance=1e-05, lambda_t=100.0, verbose=False):
    c0_s = N.linspace(x.min(), x.max(), num=10)
    best = None
    for i in range(1, (len(c0_s) - 1)):
        try:
            p = fit__zero_mean(x, y, a0=y.max(), c0=c0_s[i], tolerance=tolerance, lambda_t=lambda_t, verbose=verbose)
            if (best is None) or (best['e'] > p['e']):
                best = p
        except N.linalg.LinAlgError:
            pass
    return best
