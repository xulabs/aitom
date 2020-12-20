"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import os
import sys
import copy
import time
import traceback
import warnings
import uuid
import pickle as pickle
import shutil
from collections import defaultdict
import scipy.stats as SCS
from multiprocessing.pool import Pool as Pool
import numpy as N
import numpy.fft as NF
from aitom.tomominer.io.cache import Cache
from aitom.tomominer.common.obj import Object
import aitom.tomominer.dimension_reduction.util as DU
import aitom.image.vol.util as UV
import aitom.tomominer.io.file as IV
import aitom.geometry.ang_loc as AAL
import aitom.geometry.rotate as GR
import aitom.tomominer.align.fast.full as AFF
import aitom.tomominer.align.refine.gradient_refine as AFGF
import aitom.align.fast.util as AU
import aitom.tomominer.statistics.ssnr as SS
import aitom.tomominer.segmentation.watershed as SW
import aitom.tomominer.segmentation.active_contour.chan_vese.segment as SACS
import aitom.tomominer.filter.gaussian as FG
import aitom.tomominer.model.util as MU
import aitom.tomominer.core.core as core


def align_vols_with_wedge(v1, m1, v2, m2, op=None, logger=None):
    err = None
    if 'fast_align_and_refine' in op:
        try:
            faar_op = copy.deepcopy(op['fast_align_and_refine'])
            faar_op['fast_max_l'] = op['L']
            faar_op.update({'v1': v1, 'm1': m1, 'v2': v2, 'm2': m2, })
            re = AFGF.fast_align_and_refine(faar_op)
            ang = re['ang']
            loc = re['loc_r']
            score = re['cor']
        except Exception as e:
            err = traceback.format_exc()
    else:
        try:
            re = AU.align_vols(v1=v1, m1=m1, v2=v2, m2=m2, L=op['L'])
            ang = re['angle']
            loc = re['loc']
            score = re['score']
        except Exception as e:
            err = traceback.format_exc()
    if err is not None:
        score = float('nan')
        loc = N.zeros(3)
        ang = (N.random.random(3) * (N.pi * 2))
    return {'angle': ang, 'loc': loc, 'score': score, 'err': err, }


def align_keys(self, v1k, v2k, op, v2_out_key=None):
    if self is None:
        self = Object()
        self.cache = Cache()
    if op['with_missing_wedge']:
        v1 = self.cache.get_mrc(v1k['subtomogram'])
        if 'mask' in v1k:
            m1 = self.cache.get_mrc(v1k['mask'])
        else:
            m1 = None
        v2 = self.cache.get_mrc(v2k['subtomogram'])
        if 'mask' in v2k:
            m2 = self.cache.get_mrc(v2k['mask'])
        else:
            m2 = None
        re = align_vols_with_wedge(v1, m1, v2, m2, op=op)
    else:
        v1 = self.cache.get_mrc(v1k['subtomogram'])
        v2 = self.cache.get_mrc(v2k['subtomogram'])
        re = align_vols_no_wedge(v1=v1, v2=v2, op=op)
    re['v1_key'] = v1k
    re['v2_key'] = v2k
    re['v2_out_key'] = v2_out_key
    if v2_out_key is not None:
        v2r = GR.rotate_pad_mean(v2, angle=re['angle'], loc_r=re['loc'])
        IV.put_mrc(v2r, v2_out_key['subtomogram'])
        if op['with_missing_wedge']:
            m2r = GR.rotate_mask(m2, angle=re['angle'])
            IV.put_mrc(m2r, v2_out_key['mask'])
    re['op'] = op
    return re


def rotate_key(self, vk, vk_out, angle, loc):
    if self is not None:
        v = self.cache.get_mrc(vk['subtomogram'])
    else:
        v = IV.get_mrc(vk['subtomogram'])
    vr = GR.rotate_pad_mean(v, angle=angle, loc_r=loc)
    IV.put_mrc(vr, vk_out['subtomogram'])
    if ('mask' in vk) and ('mask' in vk_out):
        if self is not None:
            m = self.cache.get_mrc(vk['mask'])
        else:
            m = IV.get_mrc(vk['mask'])
        mr = GR.rotate_mask(m, angle=angle)
        IV.put_mrc(mr, vk_out['mask'])


def impute_aligned_vols(t, v, vm, normalize=None):
    assert (normalize is not None)
    if normalize:
        v = ((v - v.mean()) / v.std())
        if t is not None:
            t = ((t - t.mean()) / t.std())
    if t is None:
        return v
    t_f = NF.fftshift(NF.fftn(t))
    v_f = NF.fftshift(NF.fftn(v))
    v_f[(vm == 0)] = t_f[(vm == 0)]
    v_f_if = N.real(NF.ifftn(NF.ifftshift(v_f)))
    if normalize:
        v_f_if = ((v_f_if - v_f_if.mean()) / v_f_if.std())
    if N.all(N.isfinite(v_f_if)):
        return v_f_if
    else:
        print('warning: imputation failed')
        return v


def impute_vols(v, vm, ang, loc, t=None, align_to_template=True, normalize=None):
    ang = N.array(ang, dtype=N.float)
    loc = N.array(loc, dtype=N.float)
    v_r = None
    vm_r = None
    t_r = None
    if align_to_template:
        v_r = GR.rotate_pad_mean(v, angle=ang, loc_r=loc)
        assert N.all(N.isfinite(v_r))
        vm_r = GR.rotate_mask(vm, angle=ang)
        assert N.all(N.isfinite(vm_r))
        vi = impute_aligned_vols(t=t, v=v_r, vm=(vm_r > 0.5), normalize=normalize)
    else:
        (ang_inv, loc_inv) = AAL.reverse_transform_ang_loc(ang, loc)
        if t is not None:
            assert N.all(N.isfinite(t))
            t_r = GR.rotate_vol_pad_mean(t, angle=ang_inv, loc_r=loc_inv)
            N.all(N.isfinite(t_r))
        vi = impute_aligned_vols(t=t_r, v=v, vm=(vm > 0.5), normalize=normalize)
    return {'vi': vi, 'v_r': v_r, 'vm_r': vm_r, 't_r': t_r, }


def impute_vol_keys(vk, ang, loc, normalize, tk=None, align_to_template=True, cache=None):
    v = cache.get_mrc(vk['subtomogram'])
    if not N.all(N.isfinite(v)):
        raise Exception('error loading', vk['subtomogram'])
    vm = cache.get_mrc(vk['mask'])
    if not N.all(N.isfinite(vm)):
        raise Exception('error loading', vk['mask'])
    t = None
    if tk is not None:
        t = IV.get_mrc(tk['subtomogram'])
        if not N.all(N.isfinite(t)):
            raise Exception('error loading', tk['subtomogram'])
    return impute_vols(v=v, vm=vm, ang=ang, loc=loc, t=t, align_to_template=align_to_template, normalize=normalize)


def neighbor_covariance_avg__parallel(self, data_json, segmentation_tg_op, normalize, n_chunk):
    start_time = time.time()
    data_json_copy = [_ for _ in data_json]
    inds = list(range(len(data_json_copy)))
    tasks = []
    while data_json_copy:
        data_json_copy_part = data_json_copy[:n_chunk]
        inds_t = inds[:n_chunk]
        tasks.append(self.runner.task(module='tomominer.pursuit.multi.util', method='neighbor_covariance__collect_info',
                                      kwargs={'data_json': data_json_copy_part,
                                              'segmentation_tg_op': segmentation_tg_op, 'normalize': normalize, }))
        data_json_copy = data_json_copy[n_chunk:]
        inds = inds[n_chunk:]
    sum_global = None
    neighbor_prod_sum = None
    for res in self.runner.run__except(tasks):
        with open(res.result) as f:
            re = pickle.load(f)
        os.remove(res.result)
        if sum_global is None:
            sum_global = re['sum']
        else:
            sum_global += re['sum']
        assert N.all(N.isfinite(sum_global))
        if neighbor_prod_sum is None:
            neighbor_prod_sum = re['neighbor_prod_sum']
        else:
            neighbor_prod_sum += re['neighbor_prod_sum']
        assert N.all(N.isfinite(neighbor_prod_sum))
    avg_global = (sum_global / len(data_json))
    neighbor_prod_avg = (neighbor_prod_sum / len(data_json))
    shift = re['shift']
    cov = N.zeros(neighbor_prod_avg.shape)
    for i in range(shift.shape[0]):
        cov[:, :, :, i] = (neighbor_prod_avg[:, :, :, i] - (
                avg_global * UV.roll(avg_global, shift[(i, 0)], shift[(i, 1)], shift[(i, 2)])))
    cov_avg = N.mean(cov, axis=3)
    print('Calculated neighbor covariance for', len(data_json), 'subtomograms',
          (' : %2.6f sec' % (time.time() - start_time)))
    return cov_avg


def neighbor_covariance__collect_info(self, data_json, segmentation_tg_op, normalize):
    sum_local = None
    neighbor_prod_sum = None
    for rec in data_json:
        if 'template' not in rec:
            rec['template'] = None
        vri = impute_vol_keys(vk=rec, ang=rec['angle'], loc=rec['loc'], tk=rec['template'], align_to_template=True,
                              normalize=normalize, cache=self.cache)['vi']
        if (segmentation_tg_op is not None) and (rec['template'] is not None) and ('segmentation' in rec['template']):
            phi = IV.read_mrc(rec['template']['segmentation'])['value']
            vri_s = template_guided_segmentation(v=vri, m=(phi > 0.5), op=segmentation_tg_op)
            if vri_s is not None:
                vri = vri_s
                del vri_s
                assert (normalize is not None)
                if normalize:
                    vri_t = N.zeros(vri.shape)
                    vri_f = N.isfinite(vri)
                    if not vri_f.sum() <= 0:
                        vri_t[vri_f] = ((vri[vri_f] - vri[vri_f].mean()) / vri[vri_f].std())
                    vri = vri_t
                    del vri_f, vri_t
                else:
                    vri_t = N.zeros(vri.shape)
                    vri_f = N.isfinite(vri)
                    vri_t[vri_f] = vri[vri_f]
                    if vri_f.sum() > 0:
                        vri_t[N.logical_not(vri_f)] = vri[vri_f].mean()
                    vri = vri_t
                    del vri_f, vri_t
        if sum_local is None:
            sum_local = vri
        else:
            sum_local += vri
        nei_prod = DU.neighbor_product(vri)
        if neighbor_prod_sum is None:
            neighbor_prod_sum = nei_prod['p']
        else:
            neighbor_prod_sum += nei_prod['p']
    re = {'sum': sum_local, 'neighbor_prod_sum': neighbor_prod_sum, 'shift': nei_prod['shift'], }
    re_key = self.cache.save_tmp_data(re, fn_id=self.task.task_id)
    assert (re_key is not None)
    return re_key


def data_matrix_collect__parallel(self, data_json, segmentation_tg_op, normalize, n_chunk, voxel_mask_inds=None):
    start_time = time.time()
    data_json_copy = [_ for _ in data_json]
    inds = list(range(len(data_json_copy)))
    tasks = []
    while data_json_copy:
        data_json_copy_t = data_json_copy[:n_chunk]
        inds_t = inds[:n_chunk]
        tasks.append(self.runner.task(module='tomominer.pursuit.multi.util', method='data_matrix_collect__local',
                                      kwargs={'data_json': data_json_copy_t, 'segmentation_tg_op': segmentation_tg_op,
                                              'normalize': normalize, 'inds': inds_t,
                                              'voxel_mask_inds': voxel_mask_inds, }))
        data_json_copy = data_json_copy[n_chunk:]
        inds = inds[n_chunk:]
    red = None
    for res in self.runner.run__except(tasks):
        with open(res.result) as f:
            re = pickle.load(f)
        os.remove(res.result)
        if red is None:
            red = N.zeros([len(data_json), re['mat'].shape[1]])
        red[re['inds'], :] = re['mat']
    print('Calculated matrix of', len(data_json), 'subtomograms', ('%2.6f sec' % (time.time() - start_time)))
    return red


def data_matrix_collect__local(self, data_json, inds, segmentation_tg_op, normalize, voxel_mask_inds=None):
    mat = None
    for (i, rec) in enumerate(data_json):
        if 'template' not in rec:
            rec['template'] = None
        vi = impute_vol_keys(vk=rec, ang=rec['angle'], loc=rec['loc'], tk=rec['template'], align_to_template=True,
                             normalize=normalize, cache=self.cache)['vi']
        if (segmentation_tg_op is not None) and (rec['template'] is not None) and ('segmentation' in rec['template']):
            phi = IV.read_mrc(rec['template']['segmentation'])['value']
            vi_s = template_guided_segmentation(v=vi, m=(phi > 0.5), op=segmentation_tg_op)
            if vi_s is not None:
                vi = vi_s
                del vi_s
                assert (normalize is not None)
                if normalize:
                    vi_t = N.zeros(vi.shape)
                    vi_f = N.isfinite(vi)
                    vi_t[vi_f] = ((vi[vi_f] - vi[vi_f].mean()) / vi[vi_f].std())
                    vi = vi_t
                    del vi_f, vi_t
                else:
                    vi_t = N.zeros(vi.shape)
                    vi_f = N.isfinite(vi)
                    vi_t[vi_f] = vi[vi_f]
                    vi_t[N.logical_not(vi_f)] = vi[vi_f].mean()
                    vi = vi_t
                    del vi_f, vi_t
        vi = vi.flatten()
        if voxel_mask_inds is not None:
            vi = vi[voxel_mask_inds]
        if mat is None:
            mat = N.zeros([len(data_json), vi.size])
        mat[i, :] = vi
    re = {'mat': mat, 'inds': inds, }
    re_key = self.cache.save_tmp_data(re, fn_id=self.task.task_id)
    assert (re_key is not None)
    return re_key


def covariance_filtered_pca(self, data_json_model=None, data_json_embed=None, normalize=None, segmentation_tg_op=None,
                            n_chunk=100, max_feature_num=None, pca_op=None):
    print('Dimension reduction')
    start_time = time.time()
    if data_json_model is None:
        assert (data_json_embed is not None)
        data_json_model = data_json_embed
        data_json_embed = None
    cov_avg = None
    cov_avg__feature_num_cutoff = None
    if (max_feature_num is None) or (max_feature_num < 0):
        voxel_mask_inds = None
    else:
        cov_avg = neighbor_covariance_avg__parallel(self=self, data_json=data_json_model,
                                                    segmentation_tg_op=segmentation_tg_op, normalize=normalize,
                                                    n_chunk=n_chunk)
        cov_avg_max = cov_avg.max()
        if (not N.isfinite(cov_avg_max)) or (cov_avg_max <= 0):
            raise Exception(('cov_avg.max(): ' + repr(cov_avg_max)))
        cov_avg_i = N.argsort((- cov_avg), axis=None)
        cov_avg__feature_num_cutoff = cov_avg.flatten()[cov_avg_i[min(max_feature_num, (cov_avg_i.size - 1))]]
        cov_avg__feature_num_cutoff = max(cov_avg__feature_num_cutoff, 0)
        voxel_mask_inds = N.flatnonzero((cov_avg > cov_avg__feature_num_cutoff))
    mat = data_matrix_collect__parallel(self=self, data_json=data_json_model, segmentation_tg_op=segmentation_tg_op,
                                        normalize=normalize, n_chunk=n_chunk, voxel_mask_inds=voxel_mask_inds)
    mat_mean = mat.mean(axis=0)
    for i in range(mat.shape[0]):
        mat[i, :] -= mat_mean
    empca_weight = N.zeros(mat.shape, dtype=float)
    empca_weight[N.isfinite(mat)] = 1.0
    if cov_avg is not None:
        cov_avg_v = cov_avg.flatten()
        for (i, ind_t) in enumerate(voxel_mask_inds):
            empca_weight[:, i] *= cov_avg_v[ind_t]
    import aitom.tomominer.dimension_reduction.empca as drempca
    pca = drempca.empca(data=mat, weights=empca_weight, nvec=pca_op['n_dims'], niter=pca_op['n_iter'])
    if (data_json_embed is None) or ():
        red = pca.coeff
    else:
        mat_embed = data_matrix_collect__parallel(self=self, data_json=data_json_embed,
                                                  segmentation_tg_op=segmentation_tg_op, normalize=normalize,
                                                  n_chunk=n_chunk, voxel_mask_inds=voxel_mask_inds)
        red = N.dot(mat_embed, pca.eigvec.T)
    print(('PCA with covariange thresholding  : %2.6f sec' % (time.time() - start_time)))
    return {'red': red, 'cov_avg': cov_avg, 'cov_avg__feature_num_cutoff': cov_avg__feature_num_cutoff,
            'voxel_mask_inds': voxel_mask_inds, }


def labels_to_clusters(data_json, labels, cluster_mode=None, ignore_negative_labels=True):
    clusters = {}
    for (l, d) in zip(labels, data_json):
        if ignore_negative_labels and (l < 0):
            continue
        if l not in clusters:
            clusters[l] = {}
        if 'cluster_mode' not in clusters[l]:
            clusters[l]['cluster_mode'] = cluster_mode
        if 'data_json' not in clusters[l]:
            clusters[l]['data_json'] = []
        clusters[l]['data_json'].append(d)
    return clusters


def kmeans_clustering(x, k):
    warnings.filterwarnings('once')
    from sklearn.cluster import KMeans
    warnings.filterwarnings('error')
    if False:
        import multiprocessing
        km = KMeans(n_clusters=k, n_jobs=multiprocessing.cpu_count())
    else:
        km = KMeans(n_clusters=k)
    labels = km.fit_predict(x)
    labels_t = (N.zeros(labels.shape, dtype=N.int) + N.nan)
    label_count = 0
    for l in N.unique(labels):
        labels_t[(labels == l)] = label_count
        label_count += 1
    labels = labels_t.astype(N.int)
    return labels


def cluster_ssnr_fsc(self, clusters, n_chunk, op=None):
    start_time = time.time()
    sps = SS.ssnr_parallel(self=self, clusters=clusters, n_chunk=n_chunk, op=op)
    return sps


def vol_avg__local(self, data_json, op=None, return_key=True):
    vol_sum = None
    mask_sum = None
    for rec in data_json:
        if self.work_queue.done_tasks_contains(self.task.task_id):
            raise Exception('Duplicated task')
        if op['with_missing_wedge'] or op['use_fft']:
            in_re = impute_vol_keys(vk=rec, ang=rec['angle'], loc=rec['loc'], tk=None, align_to_template=True,
                                    normalize=False, cache=self.cache)
            vt = in_re['v_r']
        else:
            raise Exception('following options need to be re-considered')
            if 'template' not in rec:
                rec['template'] = None
            in_re = impute_vol_keys(vk=rec, ang=rec['angle'], loc=rec['loc'], tk=rec['template'],
                                    align_to_template=True, normalize=True, cache=self.cache)
            vt = in_re['vi']
        if vol_sum is None:
            vol_sum = N.zeros(vt.shape, dtype=N.float64, order='F')
        vol_sum += vt
        if mask_sum is None:
            mask_sum = N.zeros(in_re['vm_r'].shape, dtype=N.float64, order='F')
        mask_sum += in_re['vm_r']
    re = {'vol_sum': vol_sum, 'mask_sum': mask_sum, 'vol_count': len(data_json), 'op': op, }
    if return_key:
        re_key = self.cache.save_tmp_data(re, fn_id=self.task.task_id)
        assert (re_key is not None)
        return {'key': re_key, }
    else:
        return re


def cluster_averaging_vols(self, clusters, op={}):
    start_time = time.time()
    if op['centerize_loc']:
        clusters_cnt = copy.deepcopy(clusters)
        for c in clusters_cnt:
            loc = N.zeros((len(clusters_cnt[c]), 3))
            for (i, rec) in enumerate(clusters_cnt[c]):
                loc[i, :] = N.array(rec['loc'], dtype=N.float)
            loc -= N.tile(loc.mean(axis=0), (loc.shape[0], 1))
            assert N.all((N.abs(loc.mean(axis=0)) <= 1e-10))
            for (i, rec) in enumerate(clusters_cnt[c]):
                rec['loc'] = loc[i]
        clusters = clusters_cnt
    tasks = []
    for c in clusters:
        while clusters[c]:
            part = clusters[c][:op['n_chunk']]
            op_t = copy.deepcopy(op)
            op_t['cluster'] = c
            tasks.append(self.runner.task(module='tomominer.pursuit.multi.util', method='vol_avg__local',
                                          kwargs={'data_json': part, 'op': op_t, 'return_key': True, }))
            clusters[c] = clusters[c][op['n_chunk']:]
    cluster_sums = {}
    cluster_mask_sums = {}
    cluster_sizes = {}
    for res in self.runner.run__except(tasks):
        with open(res.result['key']) as f:
            re = pickle.load(f)
        os.remove(res.result['key'])
        oc = re['op']['cluster']
        ms = re['mask_sum']
        s = re['vol_sum']
        vc = re['vol_count']
        if oc not in cluster_sums:
            cluster_sums[oc] = s
        else:
            cluster_sums[oc] += s
        if oc not in cluster_mask_sums:
            cluster_mask_sums[oc] = ms
        else:
            cluster_mask_sums[oc] += ms
        if oc not in cluster_sizes:
            cluster_sizes[oc] = vc
        else:
            cluster_sizes[oc] += vc
        del oc, ms, s, vc
    cluster_avg_dict = {}
    for c in cluster_sums:
        assert (cluster_sizes[c] > 0)
        assert (cluster_mask_sums[c].max() > 0)
        if op['use_fft']:
            ind = (cluster_mask_sums[c] >= op['mask_count_threshold'])
            if ind.sum() == 0:
                continue
            cluster_sums_fft = NF.fftshift(NF.fftn(cluster_sums[c]))
            cluster_avg = N.zeros(cluster_sums_fft.shape, dtype=N.complex)
            cluster_avg[ind] = (cluster_sums_fft[ind] / cluster_mask_sums[c][ind])
            cluster_avg = N.real(NF.ifftn(NF.ifftshift(cluster_avg)))
        else:
            cluster_avg = (cluster_sums[c] / cluster_sizes[c])
        if op['mask_binarize']:
            cluster_mask_avg = (cluster_mask_sums[c] >= op['mask_count_threshold'])
        else:
            cluster_mask_avg = (cluster_mask_sums[c] / cluster_sizes[c])
        cluster_avg_dict[c] = {'vol': cluster_avg, 'mask': cluster_mask_avg, }
    if 'smooth' in op:
        print('smoothing', op['smooth']['mode'], end=' ')
        for c in cluster_avg_dict:
            if c not in op['smooth']['fsc']:
                continue
            cluster_avg_dict[c]['smooth'] = {'vol-original': cluster_avg_dict[c]['vol'], }
            s = cluster_averaging_vols__smooth(v=cluster_avg_dict[c]['vol'], fsc=op['smooth']['fsc'][c],
                                               mode=op['smooth']['mode'])
            cluster_avg_dict[c]['vol'] = s['v']
            if 'fit' in s:
                cluster_avg_dict[c]['smooth']['fit'] = s['fit']
                print('average', c, 'sigma ', cluster_avg_dict[c]['smooth']['fit']['c'], '    ', end=' ')
        print()
    return {'cluster_avg_dict': cluster_avg_dict, 'cluster_sums': cluster_sums, 'cluster_mask_sums': cluster_mask_sums,
            'cluster_sizes': cluster_sizes, }


def cluster_averaging_vols__smooth(v, fsc, mode):
    re = {}
    assert N.all((fsc >= 0))
    if not fsc.max() != 0.0:
        return {'v': v, }
    if mode == 'fsc_direct':
        band_pass_curve = fsc
    elif mode == 'fsc_gaussian':
        import aitom.tomominer.fitting.gaussian.one_dim as FGO
        bands = N.array(list(range(len(fsc))))
        fit = FGO.fit__zero_mean__multi_start(x=bands, y=fsc)
        if fit['c'] < bands.max():
            re['fit'] = fit
            band_pass_curve = FGO.fit__zero_mean__gaussian_function(x=bands, a=fit['a'], c=fit['c'])
        else:
            re['v'] = v
            return re
    else:
        raise Exception('mode')
    import aitom.tomominer.filter.band_pass as IB
    re['v'] = IB.filter_given_curve(v=v, curve=band_pass_curve)
    return re


def cluster_averaging(self, clusters, op={}):
    cav = cluster_averaging_vols(self, clusters=clusters, op=op)
    if not os.path.isdir(op['out_dir']):
        os.makedirs(op['out_dir'])
    with open(os.path.join(op['out_dir'], 'cluster.pickle'), 'wb') as f:
        pickle.dump(cav, f, protocol=(-1))
    cluster_avg_dict = cav['cluster_avg_dict']
    template_keys = {}
    for c in cluster_avg_dict:
        template_keys[c] = {}
        template_keys[c]['cluster'] = c
        vol_avg_out_key = os.path.join(op['out_dir'], ('clus_vol_avg_%03d.mrc' % c))
        IV.put_mrc(N.array(cluster_avg_dict[c]['vol'], order='F'), vol_avg_out_key)
        template_keys[c]['subtomogram'] = vol_avg_out_key
        if 'smooth' in cluster_avg_dict[c]:
            vol_avg_original_out_key = os.path.join(op['out_dir'], ('clus_vol_avg_original_%03d.mrc' % c))
            IV.put_mrc(N.array(cluster_avg_dict[c]['smooth']['vol-original'], order='F'), vol_avg_original_out_key)
            template_keys[c]['subtomogram-original'] = vol_avg_original_out_key
        else:
            vol_avg__riginal_out_key = None
        mask_avg_out_key = os.path.join(op['out_dir'], ('clus_mask_avg_%03d.mrc' % c))
        IV.put_mrc(N.array(cluster_avg_dict[c]['mask'], order='F'), mask_avg_out_key)
        template_keys[c]['mask'] = mask_avg_out_key
        if 'pass_i' in op:
            template_keys[c]['pass_i'] = op['pass_i']
    return {'template_keys': template_keys, }


def cluster_average_select_fsc(self, cluster_info, cluster_info_stat, op=None, debug=False):
    ci = []
    for i in cluster_info:
        for c in cluster_info[i]:
            if 'fsc' not in cluster_info[i][c]:
                continue
            if len(cluster_info[i][c]['data_json']) < op['cluster']['size_min']:
                continue
            if (not op['keep_non_specific_clusters']) and ('is_specific' in cluster_info_stat[i][c]) and (
                    cluster_info_stat[i][c]['is_specific'] is not None):
                continue
            ci.append(cluster_info[i][c])
    ci = sorted(ci, key=(lambda x: float((- x['fsc'].sum()))))
    ci_cover = []
    covered_set = set()
    for ci_t in ci:
        if 'template_key' not in ci_t:
            continue
        subtomograms_t = set((_['subtomogram'] for _ in ci_t['data_json']))
        overlap_ratio_t = (float(len(covered_set.intersection(subtomograms_t))) / len(subtomograms_t))
        if overlap_ratio_t <= op['cluster']['overlap_ratio']:
            ci_cover.append(ci_t)
            covered_set.update(subtomograms_t)
        if debug:
            print(ci_t['pass_i'], ci_t['cluster'], len(ci_t['data_json']), ci_t['fsc'].sum(), overlap_ratio_t)
    del ci
    print('Set sizes', sorted([len(_['data_json']) for _ in ci_cover]))
    tk = {}
    for (i, cc) in enumerate(ci_cover):
        tk[i] = copy.deepcopy(cc['template_key'])
        tk[i]['id'] = i
        assert (tk[i]['pass_i'] == cc['pass_i'])
        assert (tk[i]['cluster'] == cc['cluster'])
    tk_selected = set((tk[_]['subtomogram'] for _ in tk))
    tk_info = {}
    for i in cluster_info:
        for c in cluster_info[i]:
            if 'template_key' not in cluster_info[i][c]:
                continue
            tk_subtomogram = cluster_info[i][c]['template_key']['subtomogram']
            if tk_subtomogram not in tk_selected:
                continue
            tk_info[tk_subtomogram] = cluster_info[i][c]
    if op['keep_non_specific_clusters']:
        for (i, tkt) in tk.items():
            cluster_info_stat[tkt['pass_i']][tkt['cluster']]['is_specific'] = None
    assert (len(tk) > 0)
    return {'selected_templates': tk, 'tk_info': tk_info, }


def cluster_removal_according_to_center_matching_specificity(ci, cis, al, tk, significance_level, test_type=0,
                                                             test_sample_num_min=10):
    tk = copy.deepcopy(tk)
    tkd = {tk[_]['subtomogram']: _ for _ in tk}
    tk_fsc = {}
    for pass_i in ci:
        for c in ci[pass_i]:
            ci0 = ci[pass_i][c]
            if 'template_key' not in ci0:
                continue
            tk0 = ci0['template_key']['subtomogram']
            if tk0 not in tkd:
                continue
            tk_fsc[tk0] = ci0['fsc'].sum()
    non_specific_clusters = []
    wilcoxion_stat = defaultdict(dict)
    for pass_i in ci:
        for ci_c0 in ci[pass_i]:
            ci0 = ci[pass_i][ci_c0]
            cis0 = cis[pass_i][ci_c0]
            if 'is_specific' not in cis0:
                cis0['is_specific'] = None
            if cis0['is_specific'] is not None:
                continue
            if 'template_key' not in ci0:
                continue
            tk0 = ci0['template_key']['subtomogram']
            if tk0 not in tkd:
                continue
            c0 = tkd[tk0]
            ci0s = set((str(_['subtomogram']) for _ in ci0['data_json']))
            al0 = [_ for _ in al if (_['vol_key']['subtomogram'] in ci0s)]
            ss = {}
            for c1 in tk:
                tk1 = tk[c1]['subtomogram']
                if tk1 not in tkd:
                    continue
                if tk_fsc[tk0] > tk_fsc[tk1]:
                    continue
                ss[c1] = N.array([_['align'][c1]['score'] for _ in al0])
            for c1 in ss:
                if c1 == c0:
                    continue
                tk1 = tk[c1]['subtomogram']
                assert (tk1 is not tk0)
                assert (tk1 in tkd)
                assert (tk_fsc[tk1] > tk_fsc[tk0])
                ind_t = N.logical_and(N.isfinite(ss[c0]), N.isfinite(ss[c1]))
                if ind_t.sum() < test_sample_num_min:
                    continue
                if N.all((ss[c0][ind_t] > ss[c1][ind_t])):
                    continue
                is_specific = None
                if 0 == test_type:
                    (t_, p_) = SCS.wilcoxon(ss[c1][ind_t], ss[c0][ind_t])
                    if p_ > significance_level:
                        is_specific = {'tk0': tk[c0], 'tk1': tk[c1], 'stat': {'t': t_, 'p': p_, }, }
                    elif N.median(ss[c1][ind_t]) > N.median(ss[c0][ind_t]):
                        is_specific = {'tk0': tk[c0], 'tk1': tk[c1], 'stat': {'t': t_, 'p': p_, }, }
                elif test_type == 1:
                    (t_, p_) = SCS.mannwhitneyu(ss[c1][ind_t], ss[c0][ind_t])
                    if p_ >= significance_level:
                        is_specific = {'tk0': tk[c0], 'tk1': tk[c1], 'stat': {'t': t_, 'p': p_, }, }
                else:
                    raise AttributeError('test_type')
                wilcoxion_stat[c0][c1] = {'t': t_, 'p': p_, 'median_c0': N.median(ss[c0][ind_t]),
                                          'median_c1': N.median(ss[c1][ind_t]), }
                if is_specific is not None:
                    cis0['is_specific'] = is_specific
                    del tkd[tk0]
                    non_specific_clusters.append(ci0)
                    break
    none_specific_cluster_ids = []
    for c in list(tk.keys()):
        if tk[c]['subtomogram'] in tkd:
            continue
        del tk[c]
        none_specific_cluster_ids.append(c)
    for al_ in al:
        best = {'score': (- N.inf),
                'template_id': None,
                'angle': (N.random.random(3) * (N.pi * 2)),
                'loc': N.zeros(3)}
        for c in tk:
            al_c = al_['align'][c]
            if al_c['score'] > best['score']:
                best['score'] = al_c['score']
                best['angle'] = al_c['angle']
                best['loc'] = al_c['loc']
                best['template_id'] = c
        al_['best'] = best
    print(len(non_specific_clusters), 'redundant averages detected', none_specific_cluster_ids)
    sys.stdout.flush()
    return {'non_specific_clusters': non_specific_clusters, 'wilcoxion_stat': wilcoxion_stat, }


def cluster_average_align_common_frame__pairwise_alignment(self, template_keys, align_op):
    template_keys_inv = {}
    for c in template_keys:
        template_keys_inv[template_keys[c]['subtomogram']] = c
    tasks = []
    for c0 in template_keys:
        for c1 in template_keys:
            if c1 <= c0:
                continue
            tasks.append(self.runner.task(module='tomominer.pursuit.multi.util', method='align_keys',
                                          kwargs={'v1k': template_keys[c0], 'v2k': template_keys[c1],
                                                  'op': align_op, }))
    pair_align = defaultdict(dict)
    for res_t in self.runner.run__except(tasks):
        res = res_t.result
        c0 = template_keys_inv[res['v1_key']['subtomogram']]
        c1 = template_keys_inv[res['v2_key']['subtomogram']]
        pair_align[c0][c1] = res
        pair_align[c0][c1].update({'c0': c0, 'c1': c1, })
        pair_align[c1][c0] = copy.deepcopy(res)
        (ang_rev, loc_rev) = AAL.reverse_transform_ang_loc(ang=res['angle'], loc_r=res['loc'])
        pair_align[c1][c0].update({'angle': ang_rev, 'loc': loc_rev, 'c0': c1, 'c1': c0, })
        if res['err'] is not None:
            print(('cluster_average_align_common_frame__pairwise() alignment error ' + repr(res['err'])))
    return pair_align


def cluster_average_align_common_frame__multi_pair(self, tk, align_op, loc_r_max, pass_dir):
    print('align averages to common frames')
    out_dir = os.path.join(pass_dir, 'common_frame')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    pa = cluster_average_align_common_frame__pairwise_alignment(self=self, template_keys=tk, align_op=align_op)
    pal = []
    for c0 in pa:
        for c1 in pa[c0]:
            if c0 >= c1:
                continue
            if N.linalg.norm(pa[c0][c1]['loc']) > loc_r_max:
                continue
            pal.append(pa[c0][c1])
    pal = sorted(pal, key=(lambda _: (- _['score'])))
    align_to_clus = {}
    unrotated_clus = set()
    for i in range(len(pal)):
        palt = pal[i]
        c0 = palt['c0']
        c1 = palt['c1']
        assert (c0 < c1)
        if c1 in unrotated_clus:
            continue
        if c0 in align_to_clus:
            continue
        if c1 in align_to_clus:
            continue
        unrotated_clus.add(c0)
        align_to_clus[c1] = c0
    assert (len(unrotated_clus.intersection(list(align_to_clus.keys()))) == 0)
    tka = {}
    for c in tk:
        tka[c] = {}
        tka[c]['id'] = c
        tka[c]['pass_i'] = tk[c]['pass_i']
        tka[c]['cluster'] = tk[c]['cluster']
        tka[c]['subtomogram'] = os.path.join(out_dir, ('clus_vol_avg_%03d.mrc' % (c,)))
        tka[c]['mask'] = os.path.join(out_dir, ('clus_mask_avg_%03d.mrc' % (c,)))
        if c in align_to_clus:
            rotate_key(self=self, vk=tk[c], vk_out=tka[c], angle=pa[align_to_clus[c]][c]['angle'],
                       loc=pa[align_to_clus[c]][c]['loc'])
            print(align_to_clus[c], '-', c, ':', ('%0.3f' % (pa[align_to_clus[c]][c]['score'],)),
                  N.linalg.norm(pa[align_to_clus[c]][c]['loc']), '\t', end=' ')
        else:
            shutil.copyfile(tk[c]['subtomogram'], tka[c]['subtomogram'])
            shutil.copyfile(tk[c]['mask'], tka[c]['mask'])
            print(('copy(%d)' % (c,)), '\t', end=' ')
    print()
    sys.stdout.flush()
    return {'tka': tka, 'unrotated_clus': unrotated_clus, 'align_to_clus': align_to_clus, 'pa': pa, 'pal': pal, }


def template_segmentation__single(c, tk, op):
    out_path = os.path.join(os.path.dirname(tk['subtomogram']), ('clus_vol_seg_phi_%03d.mrc' % (c,)))
    v = IV.read_mrc(tk['subtomogram'])['value']
    phi = template_segmentation__single_vol(v=v, op=op)
    if phi is not None:
        tk['segmentation'] = out_path
        IV.put_mrc(phi, out_path, overwrite=True)
    return {'c': c, 'tk': tk, }


def template_segmentation__single_vol(v, op):
    if not op['density_positive']:
        v = (- v)
    del op['density_positive']
    if ('normalize_and_take_abs' in op) and op['normalize_and_take_abs']:
        v -= v.mean()
        v = N.abs(v)
    if 'gaussian_smooth_sigma' in op:
        vg = FG.smooth(v=v, sigma=float(op['gaussian_smooth_sigma']))
        del op['gaussian_smooth_sigma']
    else:
        vg = v
    phi = SACS.segment_with_postprocessing(vg, op)
    if phi is None:
        sys.stderr.write((('Warning: segmentation failed ' + out_path) + '\n'))
    if phi is not None:
        bm = MU.boundary_mask(phi.shape)
        if bm[(phi < 0)].sum() < bm[(phi > 0)].sum():
            phi = None
            sys.stderr.write((('Warning: segmentation of the following cluster average violates '
                               'boundary condition ' + out_path) + '\n'))
    return phi


def template_segmentation(self, tk, op, multiprocessing=False):
    if multiprocessing:
        if self.pool is None:
            self.pool = Pool()
        pool_results = [
            self.pool.apply_async(func=template_segmentation__single, kwds={'c': c, 'tk': tk[c], 'op': op, }) for c in
            tk]
        for r in pool_results:
            r = r.get(999999)
            tk[r['c']] = r['tk']
        self.pool.close()
        self.pool = None
    else:
        for c in tk:
            r = template_segmentation__single(c=c, tk=copy.deepcopy(tk[c]), op=copy.deepcopy(op))
            tk[r['c']] = r['tk']


def template_guided_segmentation(v, m, op):
    op = copy.deepcopy(op)
    v_org = N.copy(v)
    if not op['density_positive']:
        v = (- v_org)
    del op['density_positive']
    if 'gaussian_smooth_sigma' in op:
        vg = FG.smooth(v=v, sigma=float(op['gaussian_smooth_sigma']))
        del op['gaussian_smooth_sigma']
    else:
        vg = v
    if (m > 0.5).sum() == 0:
        return
    if (m < 0.5).sum() == 0:
        return
    op['mean_values'] = [vg[(m < 0.5)].mean(), vg[(m > 0.5)].mean()]
    phi = SACS.segment_with_postprocessing(vg, op)
    if phi is None:
        return
    if 0 == (phi > 0).sum():
        return
    if (phi < 0).sum() == 0:
        return
    struc = N.array((phi > 0), dtype=N.int32, order='F')
    mcr = core.connected_regions(struc)
    struc_tem = N.zeros(vg.shape)
    for l in range(1, (mcr['max_lbl'] + 1)):
        if N.logical_and((mcr['lbl'] == l), m).sum() > 0:
            struc_tem[(mcr['lbl'] == l)] = 1
        else:
            struc_tem[(mcr['lbl'] == l)] = 2
    if (struc_tem == 1).sum() == 0:
        return
    sws = SW.segment(vol_map=phi, vol_lbl=struc_tem)
    seg = (sws['vol_seg_lbl'] == 1)
    if seg.sum() == 0:
        return
    seg = N.logical_and((phi > (phi[seg].max() * op['phi_propotion_cutoff'])), seg)
    if seg.sum() == 0:
        return
    vs = (N.zeros(v.shape) + N.nan)
    vs[seg] = v_org[seg]
    return vs


def align_to_templates__pair_align(c, t_key, v, vm, align_op):
    if align_op['with_missing_wedge']:
        t = IV.get_mrc(t_key['subtomogram'])
        tm = IV.get_mrc(t_key['mask'])
        at_re = align_vols_with_wedge(v1=t, m1=tm, v2=v, m2=vm, op=align_op)
    else:
        t = IV.get_mrc(t_key['subtomogram'])
        at_re = align_vols_no_wedge(v1=t, v2=vi, op=align_op)
    at_re['c'] = c
    return at_re


def align_to_templates(self, rec=None, segmentation_tg_op=None, tem_keys=None, template_wedge_cutoff=0.1, align_op=None,
                       multiprocessing=False):
    vi = None
    if align_op['with_missing_wedge']:
        v = self.cache.get_mrc(rec['subtomogram'])
        vm = self.cache.get_mrc(rec['mask'])
    else:
        raise Exception('following options are need to be doube checked')
        if 'template' not in rec:
            rec['template'] = None
        vi = impute_vol_keys(vk=rec, ang=rec['angle'], loc=rec['loc'], tk=rec['template'], align_to_template=False,
                             normalize=True, cache=self.cache)['vi']
    if (segmentation_tg_op is not None) and ('template' in rec) and ('segmentation' in rec['template']):
        v = align_to_templates__segment(rec=rec, v=v, segmentation_tg_op=segmentation_tg_op)['v']
    if multiprocessing:
        if self.pool is None:
            self.pool = Pool()
        pool_results = [self.pool.apply_async(func=align_to_templates__pair_align,
                                              kwds={'c': c, 't_key': tem_keys[c], 'v': v, 'vm': vm,
                                                    'align_op': align_op, }) for c in tem_keys]
        align_re = {}
        for r in pool_results:
            at_re = r.get(999999)
            c = at_re['c']
            align_re[c] = at_re
            if N.isnan(align_re[c]['score']):
                if self.logger is not None:
                    self.logger.warning('alignment failed: rec %s, template %s, error %s ', repr(rec),
                                        repr(tem_keys[c]), repr(align_re[c]['err']))
        self.pool.close()
        self.pool = None
    else:
        align_re = {}
        for c in tem_keys:
            if self.work_queue.done_tasks_contains(self.task.task_id):
                raise Exception('Duplicated task')
            align_re[c] = align_to_templates__pair_align(c=c, t_key=tem_keys[c], v=v, vm=vm, align_op=align_op)
            if N.isnan(align_re[c]['score']):
                if self.logger is not None:
                    self.logger.warning('alignment failed: rec %s, template %s, error %s ', repr(rec),
                                        repr(tem_keys[c]), repr(align_re[c]['err']))
    return {'vol_key': rec, 'align': align_re, }


def align_to_templates__segment(rec, v, segmentation_tg_op):
    phi = IV.read_mrc_vol(rec['template']['segmentation'])
    phi_m = (phi > 0.5)
    (ang_inv, loc_inv) = AAL.reverse_transform_ang_loc(rec['angle'], rec['loc'])
    phi_mr = GR.rotate(phi_m, angle=ang_inv, loc_r=loc_inv, default_val=0)
    v_s = template_guided_segmentation(v=v, m=phi_mr, op=segmentation_tg_op)
    if (v_s is not None) and (v_s[N.isfinite(v_s)].std() > 0):
        v = v_s
        del v_s
        v_t = N.zeros(v.shape)
        v_f = N.isfinite(v)
        v_t[v_f] = ((v[v_f] - v[v_f].mean()) / v[v_f].std())
        v = v_t
        del v_f, v_t
    return {'v': v, 'phi_m': phi_m, 'phi_mr': phi_mr, }


def align_to_templates__batch(self, op, data_json, segmentation_tg_op, tmp_dir, tem_keys):
    if ('template' in op) and ('match' in op['template']) and ('priority' in op['template']['match']):
        task_priority = op['template']['match']['priority']
    else:
        task_priority = (2000 + N.random.randint(100))
    print('align against templates', 'segmentation_tg_op',
          (segmentation_tg_op if op['template']['match']['use_segmentation_mask'] else None), 'task priority',
          task_priority)
    sys.stdout.flush()
    at_ress = []
    for f in os.listdir(tmp_dir):
        if not f.endswith('.pickle'):
            continue
        res_file = os.path.join(tmp_dir, f)
        if not os.path.isfile(res_file):
            continue
        with open(res_file, 'rb') as f:
            at_ress_t = pickle.load(f)
        at_ress.append(at_ress_t)
    if len(at_ress) > 0:
        print('loaded previous', len(at_ress), ' resutlts')
        sys.stdout.flush()
    completed_subtomogram_set = set([_.result['vol_key']['subtomogram'] for _ in at_ress])
    tasks = []
    for rec in data_json:
        if rec['subtomogram'] in completed_subtomogram_set:
            continue
        tasks.append(
            self.runner.task(priority=task_priority, module='tomominer.pursuit.multi.util', method='align_to_templates',
                             kwargs={'rec': rec, 'segmentation_tg_op': (
                                 segmentation_tg_op if op['template']['match']['use_segmentation_mask'] else None),
                                     'tem_keys': tem_keys, 'align_op': op['align'], 'multiprocessing': False, }))
    for at_ress_t in self.runner.run__except(tasks):
        at_ress.append(at_ress_t)
        res_file = os.path.join(tmp_dir, ('%s.pickle' % at_ress_t.task_id))
        with open(res_file, 'wb') as f:
            pickle.dump(at_ress_t, f, protocol=0)
    return at_ress


def cluster_formation_alignment_fsc__by_global_maximum(self, dj, op=None):
    if 'debug' not in op:
        op['debug'] = False
    dj = copy.deepcopy(dj)
    djm = defaultdict(list)
    for r in dj:
        if 'template' not in r:
            continue
        djm[str(r['template']['subtomogram'])].append(r)
    djm_org = copy.deepcopy(djm)
    for k in djm:
        djmt = djm[k]
        djmt = sorted(djmt, key=(lambda _: float(_['score'])), reverse=True)
        if ('max_expansion_size' in op) and (len(djmt) > op['max_expansion_size']):
            djmt = djmt[:op['max_expansion_size']]
        djm[k] = djmt
    ssnr_sequential_op = copy.deepcopy(op['ssnr_sequential'])
    ssnr_sequential_op['n_chunk'] = op['n_chunk']
    ssnr_s = SS.ssnr_sequential_parallel(self=self, data_json_dict=djm, op=ssnr_sequential_op)
    fsc_sum = {}
    for k in ssnr_s:
        fsc_sum[k] = N.array([N.sum(_) for _ in ssnr_s[k]['fsc']])
    import scipy.ndimage.filters as SDF
    if 'gaussian_smooth_sigma' in op:
        for k in fsc_sum:
            fsc_sum[k] = SDF.gaussian_filter1d(fsc_sum[k], op['gaussian_smooth_sigma'])
    if 'min_expansion_size' in op:
        for k in copy.deepcopy(list(fsc_sum.keys())):
            if len(fsc_sum[k]) < op['min_expansion_size']:
                del fsc_sum[k]
                continue
            fsc_sum[k][:op['min_expansion_size']] = (N.min(fsc_sum[k]) - 1)
    dj_gm = {}
    for k in fsc_sum:
        i = N.argmax(fsc_sum[k])
        if op['debug']:
            print('template', k, 'original subtomogram num', len(djm_org[k]), 'global maximum', i)
        dj_gm[k] = {'k': k, 'i': i, 'data_json': copy.deepcopy(djm[k][:(i + 1)]), 'fsc': ssnr_s[k]['fsc'][i],
                    'fsc_sum': fsc_sum[k][i], }
    return {'dj_gm': dj_gm, 'djm': djm, 'ssnr_s': ssnr_s, }
