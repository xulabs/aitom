"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import os, sys, json, random
import numpy as N
import sklearn.cluster as SC
import aitom.tomominer.io.file as IF
import aitom.geometry.rotate as GR
import warnings


def process(op):
    with open(op['input data json file']) as f:
        dj = json.load(f)
    if 'test' in op:
        if ('sample_num' in op['test']) and (op['test']['sample_num'] > 0) and (len(dj) > op['test']['sample_num']):
            print(('testing the procedure using a subsample of %d subtomograms' % op['test']['sample_num']))
            dj = random.sample(dj, op['test']['sample_num'])
    mat = None
    for (i, d) in enumerate(dj):
        print('\rloading', i, '            ', end=' ')
        sys.stdout.flush()
        v = IF.read_mrc_vol(d['subtomogram'])
        if op['mode'] == 'pose':
            vr = GR.rotate_pad_mean(v, rm=N.array(d['pose']['rm']), c1=N.array(d['pose']['c']))
        elif op['mode'] == 'template':
            vr = GR.rotate_pad_mean(v, angle=N.array(d['angle']), loc_r=N.array(d['loc']))
        else:
            raise Exception('op[mode]')
        if mat is None:
            mat = N.zeros((len(dj), vr.size))
        mat[i, :] = vr.flatten()
    if 'PCA' in op:
        import aitom.tomominer.dimension_reduction.empca as drempca
        pca = drempca.empca(data=mat, weights=N.ones(mat.shape), nvec=op['PCA']['n_dims'], niter=op['PCA']['n_iter'])
        mat_km = pca.coeff
    else:
        mat_km = mat
    km = SC.KMeans(n_clusters=op['kmeans']['cluster num'], n_init=op['kmeans']['n_init'],
                   n_jobs=(op['kmeans']['n_jobs'] if ('n_jobs' in op['kmeans']) else (-1)),
                   verbose=op['kmeans']['verbose'])
    lbl = km.fit_predict(mat_km)
    dj_new = []
    for (i, d) in enumerate(dj):
        dn = {}
        if 'id' in d:
            dn['id'] = d['id']
        dn['subtomogram'] = d['subtomogram']
        dn['cluster_label'] = int(lbl[i])
        dj_new.append(dn)
    op['output data json file'] = os.path.abspath(op['output data json file'])
    if not os.path.isdir(os.path.dirname(op['output data json file'])):
        os.makedirs(os.path.dirname(op['output data json file']))
    with open(op['output data json file'], 'w') as f:
        json.dump(dj_new, f, indent=2)
    clus_dir = os.path.join(op['out dir'], 'vol-avg')
    if not os.path.isdir(clus_dir):
        os.makedirs(clus_dir)
    clus_stat = []
    for l in set(lbl.tolist()):
        avg_file_name = os.path.abspath(os.path.join(clus_dir, ('%03d.mrc' % (l,))))
        v_avg = mat[(lbl == l), :].sum(axis=0).reshape(v.shape)
        IF.put_mrc(mrc=v_avg, path=avg_file_name, overwrite=True)
        clus_stat.append(
            {'cluster_label': l, 'size': len([_ for _ in lbl if (_ == l)]), 'subtomogram': avg_file_name, })
    op['output cluster stat file'] = os.path.abspath(op['output cluster stat file'])
    if not os.path.isdir(os.path.dirname(op['output cluster stat file'])):
        os.makedirs(os.path.dirname(op['output cluster stat file']))
    with open(op['output cluster stat file'], 'w') as f:
        json.dump(clus_stat, f, indent=2)
