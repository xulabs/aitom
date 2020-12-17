"""
calculate normalized cross correlation between a template (masked so that the values outside the mask are NaN),
at every position of a map
"""

import numpy as N
from . import convolve as FC


def cor(t, v):
    """
    parameters: t:template (NaN values are outside the mask), v:map
    derivation
    {1 \over n} \sum_i[x_i - {1 \over n}\sum_j(x_j)]^2\\
    ={1 \over n} \sum_i[x_i^2 - 2x_i{1 \over n}\sum_j(x_j) + {1 \over n^2}(\sum_j x_j)^2]\\
    ={1 \over n} \sum_i x_i^2 - {2 \over n^2}\sum_i x_i \sum_j x_j + {1 \over n^2}(\sum_j x_j)^2 \\
    = {1 \over n} \sum_i x_i^2 - ({1 \over n}\sum_j x_j)^2
    """
    # mask
    m = N.isfinite(t)
    n = m.sum()

    t -= t[m].mean()
    t /= t[m].std()

    m = m.astype(N.float)

    # masked mean of v
    s_mean = FC.convolve(t=m, v=v) / n
    # masked mean of square of v
    ss_mean = FC.convolve(t=m, v=N.square(v)) / n

    v_std = ss_mean - N.square(s_mean)
    v_std[v_std < 0.0] = 0.0
    v_std = N.sqrt(v_std)

    ind = v_std > 0
    v = v - s_mean
    v[ind] = v[ind] / v_std[ind]

    t[N.logical_not(N.isfinite(t))] = 0.0

    return FC.convolve(t=t, v=v)
