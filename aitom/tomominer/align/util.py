

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import sys, copy
import numpy as N
import traceback
import aitom.tomominer.core.core as tomo
import aitom.tomominer.model.util as MU
import aitom.tomominer.io.file as iv

def align_vols(v1, m1, v2, m2, L):
    fail = False
    try:
        al = align_vols__multiple_rotations(v1=v1, m1=m1, v2=v2, m2=m2, L=L)
        score = al[0]['score']
        loc = al[0]['loc']
        angle = al[0]['angle']
    except Exception as err:
        print(traceback.format_exc(), file=sys.stderr)
        score = N.nan
        loc = (N.zeros(3) + N.nan)
        angle = (N.zeros(3) + N.nan)
        fail = True
    if (not N.isfinite(score)):
        fail = True
    if (len(loc) != 3):
        fail = True
    if (len(angle) != 3):
        fail = True
    if (not fail):
        return {'score': score, 'loc': loc, 'angle': angle, }
    else:
        return {'score': float('nan'), 'loc': N.zeros(3), 'angle': (N.random.random(3) * (N.pi * 2)), }

def align_vols__multiple_rotations(v1, m1, v2, m2, L):
    if (m1 is None):
        m1 = MU.sphere_mask(v1.shape)
    if (m2 is None):
        m2 = MU.sphere_mask(v2.shape)
    assert (v1.shape == m1.shape)
    assert (v2.shape == m2.shape)
    if (v1.shape != v2.shape):
        print(v1.shape)
        print(v2.shape)
        assert (v1.shape == v2.shape)
    cs = tomo.combined_search(v1.astype(N.float64), m1.astype(N.float64), v2.astype(N.float64), m2.astype(N.float64), L)
    al = ([None] * len(cs))
    for i in range(len(cs)):
        al[i] = {'score': cs[i][0], 'loc': N.copy(cs[i][1]), 'angle': N.copy(cs[i][2]), }
    al = sorted(al, key=(lambda _: _['score']), reverse=True)
    return al