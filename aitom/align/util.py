'''
Functions for subtomogram alignment
'''


import aitom.tomominer.core as tomo
import traceback
import sys

def align_vols(v1, m1, v2, m2, L=36):

    fail = False

    try:
        al = align_vols__multiple_rotations(v1=v1, m1=m1, v2=v2, m2=m2, L=L)

        # extract the score/displacement/angle from the first entry ret[0]
        score = al[0]['score']
        loc = al[0]['loc']
        angle = al[0]['angle']

    except Exception as err:
        print(traceback.format_exc(), file=sys.stderr)

        score = N.nan
        loc = N.zeros(3) + N.nan
        angle = N.zeros(3) + N.nan

        fail = True

    if not N.isfinite(score):  fail = True
    if len(loc) != 3:    fail = True
    if len(angle) != 3:    fail = True

    if not fail:
        return {'score':score, 'loc':loc, 'angle':angle}
    else:
        return {'score':float('nan'), 'loc':N.zeros(3), 'angle':N.random.random(3) * (N.pi * 2)}           # mxu: randomly assign an angle


def align_vols__multiple_rotations(v1, m1, v2, m2, L):

    if m1 is None:      m1 = MU.sphere_mask(v1.shape)
    if m2 is None:      m2 = MU.sphere_mask(v2.shape)

    assert(v1.shape == m1.shape)
    assert(v2.shape == m2.shape)
    if v1.shape != v2.shape:
        print(v1.shape)
        print(v2.shape)
        assert(v1.shape == v2.shape)

    cs = tomo.combined_search(v1.astype(N.float64), m1.astype(N.float64), v2.astype(N.float64), m2.astype(N.float64), L)

    al = [None] * len(cs)
    for i in range(len(cs)):        al[i] = {'score':cs[i][0], 'loc':N.copy(cs[i][1]), 'angle':N.copy(cs[i][2])}

    al = sorted(al, key=lambda _ : _['score'], reverse=True)            # make sure the alignment is in decreasing order

    return al


