'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''

import copy
import json
import os
import pickle as pickle

import aitom.geometry.rotate as GR
import numpy as N
import numpy.fft as NF

import aitom.tomominer.io.file as IF
import aitom.tomominer.pursuit.multi.pattern_generate.genetic_algorithm_ssnr_fsc as PMPG


def average(dj, mask_count_threshold):
    vol_sum = None
    mask_sum = None
    for d in dj:
        v = IF.read_mrc_vol(d['subtomogram'])
        if not N.all(N.isfinite(v)):
            raise Exception('error loading', d['subtomogram'])
        vm = IF.read_mrc_vol(d['mask'])
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
    ind = (mask_sum >= mask_count_threshold)
    vol_sum_fft = NF.fftshift(NF.fftn(vol_sum))
    avg = N.zeros(vol_sum_fft.shape, dtype=N.complex)
    avg[ind] = (vol_sum_fft[ind] / mask_sum[ind])
    avg = N.real(NF.ifftn(NF.ifftshift(avg)))
    return {'v': avg, 'm': (mask_sum / len(dj)), }


def main():
    with open('aligned_refine__op.json') as f:
        op = json.load(f)
    out_dir = os.path.abspath(op['out_dir'])
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(op['data_file']) as f:
        dj = json.load(f)
    for d in dj:
        if not os.path.isabs(d['subtomogram']):
            d['subtomogram'] = os.path.abspath(os.path.join(os.path.dirname(op['data_file']), d['subtomogram']))
        if not os.path.isabs(d['mask']):
            d['mask'] = os.path.abspath(os.path.join(os.path.dirname(op['data_file']), d['mask']))
    pmpg_file = os.path.join(out_dir, 'pmpg.pickle')
    if os.path.isfile(pmpg_file):
        print('loading existing result file', pmpg_file)
        with open(pmpg_file, 'rb') as f:
            pmpg = pickle.load(f)
    else:
        pmpg_dp_op = {}
        pmpg_ga_op = copy.deepcopy(op['genetic_algorithm'])
        pmpg_ga_op['sum_min'] = op['min_sample_num']
        pmpg_ga_op['evaluate']['ssnr']['mask_sum_threshold'] = op['min_sample_num']
        pmpg = {'dp': PMPG.data_prepare(dj=dj, op=pmpg_dp_op),
                'full_set': {}}
        pmpg['full_set']['evaluate'] = [
            PMPG.ga_evaluate__single(l=N.ones(len(dj)), stat=pmpg['dp'], op=pmpg_ga_op['evaluate'])]
        PMPG.ga_evaluate__scoring(pmpg['full_set']['evaluate'], op=pmpg_ga_op['evaluate']['scoring'])
        pmpg['best'] = PMPG.ga(stat=pmpg['dp'], op=pmpg_ga_op)
        pmpg['dj'] = [dj[_] for _ in range(len(dj)) if (pmpg['best']['p'][(0, _)] == 1)]
        del pmpg['dp']
        with open(pmpg_file, 'wb') as f:
            pickle.dump(pmpg, f, protocol=(-1))
        print(pmpg['full_set']['evaluate'][0]['score'])
    print('score for the full set of', len(dj), 'subtomograms:', pmpg['full_set']['evaluate'][0]['score'])
    print('score for the', pmpg['best']['p'][0, :].sum(), 'selected subtomograms:', pmpg['best']['e'][0]['score'])
    avg_re = average(dj=pmpg['dj'], mask_count_threshold=op['min_sample_num'])
    avg_dir = os.path.join(op['out_dir'], 'avg')
    if not os.path.isdir(avg_dir):
        os.makedirs(avg_dir)
    IF.put_mrc(avg_re['v'], os.path.join(avg_dir, 'vol_avg.mrc'))
    IF.put_mrc(avg_re['m'], os.path.join(avg_dir, 'mask_avg.mrc'))
    with open(os.path.join(out_dir, 'data_selected.json'), 'w') as f:
        json.dump(pmpg['dj'], f, indent=2)
