"""
Simple iterative subtomogram alignment, averaging and classification
"""

import copy
import os
import sys
import uuid

import numpy as N
import numpy.fft as NF

import aitom.geometry.rotate as GR
import aitom.io.file as AIF


def classify(dj_init=None, img_db=None, djs_file=None, avgs_file=None,
             pcas_file=None, op=None):
    """
    classify

    parameters:
        dj_init: a list of dicts, where each element looks like:
            {'subtomogram':v_id,
            'mask':mask_id,
            'angle':ang_t,
            'loc':loc_t,
            'model_id':model_id}
        img_db: a dict to find subtomogram data by its uuid (img_db[uuid] is a 3D np array)

    result(pickle file):
        average results of each class
    """
    djs = load_dict(op['data_checkpoint'])
    pcas = load_dict(op['dim_reduction']['pca']['checkpoint'])
    clus = load_dict(op['clustering']['checkpoint'])
    avgs = load_dict(op['average']['checkpoint'])

    if -1 not in djs:
        # store initial data
        assert len(djs) == 0
        djs[-1] = dj_init
        AIF.pickle_dump(djs, op['data_checkpoint'])

    dj = djs[-1]
    for pass_i in range(op['option']['pass_num']):
        if pass_i in djs:
            dj = djs[pass_i]
            continue

        # make a copy of the previous pass, for an update
        dj = copy.deepcopy(dj)

        if pass_i not in pcas:
            red = covariance_filtered_pca(dj=dj, img_db=img_db, templates=avgs,
                                          pca_op=op['dim_reduction']['pca'])
            # print(type(red))
            pcas[pass_i] = red
            AIF.pickle_dump(pcas, op['dim_reduction']['pca']['checkpoint'])
        else:
            red = copy.deepcopy(pcas[pass_i])

        if pass_i not in clus:
            lbl = kmeans_clustering(x=red, k=op['clustering']['kmeans_k'])
            clus[pass_i] = lbl
            AIF.pickle_dump(clus, op['clustering']['checkpoint'])
        else:
            lbl = clus[pass_i]
        # print('lbl', lbl)

        for d in dj:
            d['cluster'] = lbl[d['subtomogram']]

        # calculate cluster averages
        new_avgs = set()
        for c in set([lbl[_] for _ in lbl]):
            # print('c', c)
            if c in avgs:
                continue

            avg_t = vol_avg(dj=[_ for _ in dj if _['cluster'] == c],
                            op=op['average'], img_db=img_db)
            if avg_t is None:
                continue

            avgs[c] = avg_t
            avgs[c]['pass_i'] = pass_i
            avgs[c]['id'] = c

            new_avgs.add(c)

        if len(new_avgs) > 0:
            AIF.pickle_dump(avgs, op['average']['checkpoint'])

        # print('avgs')
        # for key in avgs:
        #     print('\n',pass_i,key)
        #     for key2 in avgs[key]:
        #         print(key2)
        #     print(avgs[key]['pass_i'],avgs[key]['id'])

        # re-align subtomograms
        al = align_all_pairs(avgs=avgs, dj=dj, img_db=img_db)
        a = align_all_pairs__select_best(al)
        for d in dj:
            i = d['subtomogram']
            d['loc'] = a[i]['loc']
            d['angle'] = a[i]['angle']
            d['score'] = a[i]['score']
            d['template_id'] = a[i]['template_id']

        djs[pass_i] = dj
        AIF.pickle_dump(djs, op['data_checkpoint'])


def load_dict(path):
    if not os.path.isfile(path):
        d = {}
        AIF.pickle_dump(d, path)
    else:
        d = AIF.pickle_load(path)
    return d


def covariance_filtered_pca(dj=None, img_db=None, templates=None, pca_op=None):
    """
    simplified from
    tomominer.pursuit.multi.util.covariance_filtered_pca
    """
    # print ('Dimension reduction')

    mat = data_matrix_collect(dj=dj, img_db=img_db, templates=templates)

    # centerize collected matrix, for PCA
    mat_mean = mat.mean(axis=0)
    for i in range(mat.shape[0]):
        mat[i, :] -= mat_mean

    # weight missing value regions as 0
    empca_weight = N.zeros(mat.shape, dtype=float)
    empca_weight[N.isfinite(mat)] = 1.0

    import aitom.tomominer.dimension_reduction.empca as drempca
    # note: need to watch out the R2 values to see how much variation can be explained
    # by the estimated model, if the value is small, need to increase dims
    pca = drempca.empca(data=mat, weights=empca_weight, nvec=pca_op['n_dims'],
                        niter=pca_op['n_iter'])

    red = pca.coeff

    red_d = {}
    for i, d in enumerate(dj):
        red_d[d['subtomogram']] = red[i]

    return red_d


def data_matrix_collect(dj, templates, img_db):
    """
    simplified from
    tomominer.pursuit.multi.util.data_matrix_collect__local
    """
    # print 'data_matrix_collect()'

    mat = None
    for i, d in enumerate(dj):
        v = img_db[d['subtomogram']]
        vm = img_db[d['mask']]

        # print(v.shape)
        # print(d['angle'],d['loc'])

        v_r = GR.rotate_pad_mean(v, angle=d['angle'], loc_r=d['loc'])
        assert N.all(N.isfinite(v_r))
        vm_r = GR.rotate_mask(vm, angle=d['angle'])
        assert N.all(N.isfinite(vm_r))

        if 'template' in d:
            vi = CU.impute_aligned_vols(t=templates[d['template']]['v'], v=v_r,
                                        vm=(vm_r > 0.5))
        else:
            vi = v_r

        vi = vi.flatten()

        if mat is None:
            mat = N.zeros([len(dj), vi.size])
        mat[i, :] = vi

        # print ('\r', i, '       ')
        sys.stdout.flush()

    return mat


def kmeans_clustering(x, k):
    """
    simplified from
    tomominer.pursuit.multi.util.kmeans_clustering
    """
    from sklearn.cluster import KMeans
    import multiprocessing
    km = KMeans(n_clusters=k, n_jobs=multiprocessing.cpu_count())

    # xk = x.keys()
    # In python3, type(x.keys()) is <class 'dict_keys'>!
    xk = list(x.keys())
    # print(xk,type(xk))
    assert type(xk) is list

    labels = km.fit_predict(N.array([x[_] for _ in xk]))
    labels = labels.tolist()

    # re-index labels using uuids
    labels_dict = {}
    for i, l in enumerate(labels):
        if l not in labels_dict:
            labels_dict[l] = str(uuid.uuid4())

    return {xk[_]: labels_dict[labels[_]] for _ in range(len(labels))}


def vol_avg(dj, op, img_db):
    """
    simplified from
    tomominer.pursuit.multi.util.vol_avg__local
    """
    if len(dj) < op['mask_count_threshold']:
        return None

    vol_sum = None
    mask_sum = None

    # temporary collection of local volume, and mask.
    for d in dj:
        v = img_db[d['subtomogram']]
        vm = img_db[d['mask']]

        v_r = GR.rotate_pad_mean(v, angle=d['angle'], loc_r=d['loc'])
        assert N.all(N.isfinite(v_r))
        vm_r = GR.rotate_mask(vm, angle=d['angle'])
        assert N.all(N.isfinite(vm_r))

        if vol_sum is None:
            vol_sum = N.zeros(v_r.shape, dtype=N.float64, order='F')
        vol_sum += v_r

        if mask_sum is None:
            mask_sum = N.zeros(vm_r.shape, dtype=N.float64, order='F')
        mask_sum += vm_r

    ind = mask_sum >= op['mask_count_threshold']
    if ind.sum() <= 0:
        return None

    vol_sum = NF.fftshift(NF.fftn(vol_sum))
    avg = N.zeros(vol_sum.shape, dtype=N.complex)
    avg[ind] = vol_sum[ind] / mask_sum[ind]
    avg = N.real(NF.ifftn(NF.ifftshift(avg)))

    return {'v': avg, 'm': mask_sum / float(len(dj))}


def align_all_pairs(avgs, dj, img_db, n_chunk=1000, redis_host=None):
    """
    because python variables are references, it is fine to prepare large amount of tasks, whose prameters points to a small numbers of images
    """
    # print 'align_all_pairs'

    ts = {}
    for d in dj:
        v = img_db[d['subtomogram']]
        m = img_db[d['mask']]

        for k in avgs:
            t = dict()
            t['uuid'] = str(uuid.uuid4())
            # t['module'] = 'tomominer.align.util'
            t['module'] = 'aitom.align.fast.util'
            t['method'] = 'align_vols'

            t['subtomogram_id'] = d['subtomogram']
            t['template_id'] = k

            a_t = dict()
            a_t['v1'] = avgs[k]['v']
            a_t['m1'] = avgs[k]['m']
            a_t['v2'] = v
            a_t['m2'] = m
            a_t['L'] = 36

            t['kwargs'] = a_t
            ts[t['uuid']] = t

            # al[i][k] = TAU.align_vols(v1=avgs[k]['v'], m1=avgs[k]['m'], v2=v, m2=m, L=36)
            # al[i][k]['template_id'] = k

    from collections import defaultdict
    al = defaultdict(dict)

    import aitom.parallel.multiprocessing.util as PARALLEL
    if redis_host is None:
        tr_s = [_ for _ in PARALLEL.run_iterator(ts)]
    else:
        tr_s = [_ for _ in PARALLEL.run_iterator(ts, n_chunk=n_chunk, redis_host=redis_host)]

    for tr in tr_s:
        i = tr['id']
        r = tr['result']
        al[ts[i]['subtomogram_id']][ts[i]['template_id']] = r
        al[ts[i]['subtomogram_id']][ts[i]['template_id']]['template_id'] = ts[i]['template_id']

    return al


def align_all_pairs__select_best(al):
    a = {}
    for i in al:
        km = None
        for k in al[i]:
            if km is None:
                km = k
            if al[i][k]['score'] < al[i][km]['score']:
                continue
            km = k
        a[i] = al[i][km]

    return a


def randomize_orientation(dj):
    for d in dj:
        d['loc'] = [0.0, 0.0, 0.0]
        d['angle'] = [_ for _ in N.random.random(3) * (N.pi * 2)]


def export_avgs(avgs, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for k in avgs:
        AIF.put_mrc(
            avgs[k]['v'],
            os.path.join(out_dir, '%03d--%s.mrc' % (avgs[k]['pass_i'], k)))
