"""
Functions for subtomogram alignment
"""

import aitom.tomominer.core.core as core
import traceback
import sys
import numpy as N
import aitom.model.util as MU

from numpy.fft import fftn, ifftn, fftshift, ifftshift
import numpy as N

import aitom.geometry.rotate as GR
import aitom.model.util as MU
import aitom.image.vol.util as IVU


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

    if not N.isfinite(score):
        fail = True
    if len(loc) != 3:
        fail = True
    if len(angle) != 3:
        fail = True

    if not fail:
        return {'score': score, 'loc': loc, 'angle': angle}
    else:
        # mxu: randomly assign an angle
        return {'score': float('nan'),
                'loc': N.zeros(3),
                'angle': N.random.random(3) * (N.pi * 2)}


def align_vols__multiple_rotations(v1, m1, v2, m2, L):
    if m1 is None: m1 = MU.sphere_mask(v1.shape)
    if m2 is None: m2 = MU.sphere_mask(v2.shape)

    assert (v1.shape == m1.shape)
    assert (v2.shape == m2.shape)
    if v1.shape != v2.shape:
        print(v1.shape)
        print(v2.shape)
        assert (v1.shape == v2.shape)

    cs = core.combined_search(v1.astype(N.float64), m1.astype(N.float64),
                              v2.astype(N.float64), m2.astype(N.float64), L)

    al = [{}] * len(cs)
    for i in range(len(cs)):
        al[i] = {'score': cs[i][0],
                 'loc': N.copy(cs[i][1]),
                 'angle': N.copy(cs[i][2])}

    # make sure the alignment is in decreasing order
    al = sorted(al, key=lambda x: x['score'], reverse=True)

    return al


def align_vols_no_mask(v1, v2, L=36):
    m = MU.sphere_mask(v1.shape)
    return align_vols(v1=v1, m1=m, v2=v2, m2=m, L=L)


def fast_rotation_align(v1, m1, v2, m2, max_l=36):
    """
    fast alignment according to JSB 2012 paper
    given two subtomograms and their masks, perform populate all candidate rotational angles,
    with missing wedge correction
    """
    radius = int(max(v1.shape) / 2)

    # radii must start from 1, not 0!
    radii = list(range(1, radius + 1))
    # convert to nparray
    radii = N.asarray(radii, dtype=N.float64)

    # fftshift breaks order='F'
    v1fa = abs(fftshift(fftn(v1)))
    v2fa = abs(fftshift(fftn(v2)))

    v1fa = v1fa.copy(order='F')
    v2fa = v2fa.copy(order='F')

    m1sq = N.square(m1)
    m2sq = N.square(m2)

    a1t = v1fa * m1sq
    a2t = v2fa * m2sq

    cor12 = core.rot_search_cor(a1t, a2t, radii, max_l)

    sqt_cor11 = N.sqrt(N.real(
        core.rot_search_cor(N.square(v1fa) * m1sq, m2sq, radii, max_l)))
    sqt_cor22 = N.sqrt(N.real(
        core.rot_search_cor(m1sq, N.square(v2fa) * m2sq, radii, max_l)))

    cors = cor12 / (sqt_cor11 * sqt_cor22)

    # N.real breaks order='F' by not making explicit copy.
    cors = N.real(cors)
    cors = cors.copy(order='F')

    (cor, angs) = core.local_max_angles(cors, 8)

    return angs


def translation_align__given_unshifted_fft(v1f, v2f):
    cor = fftshift(N.real(ifftn(v1f * N.conj(v2f))))

    mid_co = IVU.fft_mid_co(cor.shape)
    loc = N.unravel_index(cor.argmax(), cor.shape)

    return {'loc': (loc - mid_co), 'cor': cor[loc[0], loc[1], loc[2]]}


def translation_align_given_rotation_angles(v1, m1, v2, m2, angs):
    """
    for each angle, do a translation search
    """
    v1f = fftn(v1)
    v1f[0, 0, 0] = 0.0
    v1f = fftshift(v1f)

    a = [{}] * len(angs)
    for i, ang in enumerate(angs):
        v2r = GR.rotate_pad_mean(v2, angle=ang)
        v2rf = fftn(v2r)
        v2rf[0, 0, 0] = 0.0
        v2rf = fftshift(v2rf)

        m2r = GR.rotate_pad_zero(m2, angle=ang)
        m1_m2r = m1 * m2

        # masked images
        v1fm = v1f * m1_m2r
        v2rfm = v2rf * m1_m2r

        # normalize values
        v1fmn = v1fm / N.sqrt(N.square(N.abs(v1fm)).sum())
        v2rfmn = v2rfm / N.sqrt(N.square(N.abs(v2rfm)).sum())

        lc = translation_align__given_unshifted_fft(ifftshift(v1fmn),
                                                    ifftshift(v2rfmn))

        a[i] = {'ang': ang, 'loc': lc['loc'], 'score': lc['cor']}

    return a
