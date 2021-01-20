#!/usr/bin/env python
"""
calculate surface feature values for each voxels in a Gaussian smoothed volume, then generate a corresponding feature
strength map that can be used to filter peaks

The method is described in the following paper
A differential structure approach to membrane segmentation in electron tomography
Antonio Martinez-Sanchez, Inmaculada Garcia, Jose-Jesus Fernandez
Journal of Structural Biology 175 (2011) 372–383
"""

import numpy as N

import aitom.filter.differential as FD
import aitom.linalg.eigen as LE


def feature(vg, gradient_normalize=False):
    print('# calculate hessian matrices')
    dif = FD.diff_3d(vg)
    h = FD.hessian_3d(vg, d=dif)

    hmm = FD.hessian_3d__max_magnitude(h)
    h = FD.hessian_3d__normalize(h, hmm)

    print('# calculate eignevalues of hessian matrices')
    e = LE.eigen_value_3_symmetric_batch(h)
    for i in range(len(e)):
        e[i] = e[i] * hmm  # scale back

    del h  # save some space

    print('# calculate membrane strength')
    # see paper Martinez-Sanchez11 A differential structure approach to membrane segmentation in electron
    # tomography
    ind = e[0] < 0
    R = N.zeros(e[0].shape)
    R[ind] = N.abs(e[0][ind]) - N.sqrt(N.abs(e[1][ind] * e[2][ind]))

    del e  # save some space

    Rsq = N.square(R)
    del R

    if gradient_normalize:
        # see equation 5 of paper Martinez-Sanchez13 A ridge-based framework for segmentation of 3D electron
        # microscopy datasets
        dif_ms = FD.gradient_magnitude_square(dif)
        ind = dif_ms > 0
        M = N.zeros(Rsq.shape)
        M[ind] = Rsq[ind] / dif_ms[ind]
        return M
    else:
        return Rsq


if __name__ == '__main__':
    import json
    import aitom.io.file as IF
    import aitom.filter.gaussian as FG

    with open('surface_feature__op.json') as f:
        op = json.load(f)

    im = IF.read_mrc(op['vol_file'])
    # voxel spacing in nm unit
    voxel_spacing = (im['header']['MRC']['xlen'] / im['header']['MRC']['nx']) * 0.1
    print('voxel_spacing', voxel_spacing)

    v = im['value']

    if 'debug' in op:
        se = N.array(op['debug']['start_end'])
        v = v[se[0, 0]:se[0, 1], se[1, 0]:se[1, 1], se[2, 0]:se[2, 1]]

    v = v.astype(N.float)
    v -= v.mean()

    IF.put_mrc(v, '/tmp/v.mrc', overwrite=True)

    if op['inverse_intensity']:
        v = -v
    print('intensity distribution of v', 'max', v.max(), 'min', v.min(), 'mean', v.mean())

    print('# gaussian smoothing')

    vg = FG.smooth(v, sigma=float(op['sigma']) / voxel_spacing)

    print('intensity distribution of vg', 'max', vg.max(), 'min', vg.min(), 'mean', vg.mean())

    IF.put_mrc(vg, '/tmp/vg.mrc', overwrite=True)

    R = feature(vg)

    IF.put_mrc(R, op['out_file'], overwrite=True)

'''
    IF.put_mrc(R, '/tmp/r.mrc', overwrite=True)
    S = FD.gradient_magnitude_square(dif)
    print '# calculate directional dirvative along thee gradient'
    Rdd = FD.directional_derivative_along_gradient(R, d=dif)
    Sdd = FD.directional_derivative_along_gradient(S, d=dif)
    ind = N.logical_and((N.sign(Rdd) != N.sign(Sdd)), vg>op['intensity_cutoff'])
    print ind.sum()
    M = N.zeros(Rdd.shape)
    eps = S[ind].mean() * 1e-2
    M[ind] = N.square(R[ind]) / (S[ind] + eps)
    print M[ind].max(), M[ind].min(), M[ind].mean()
    M /= M[ind].mean()
    IF.put_mrc(M, op['out_file'], overwrite=True)
'''
