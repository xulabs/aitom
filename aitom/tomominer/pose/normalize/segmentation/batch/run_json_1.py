"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import os, sys, json, copy
from multiprocessing.pool import Pool
import numpy as N
import aitom.tomominer.pose.normalize.util as PNU
import aitom.geometry.rotate as GR
import aitom.tomominer.io.file as IF
import aitom.tomominer.filter.gaussian as FG
import aitom.tomominer.segmentation.active_contour.chan_vese.segment as SA
import traceback


def level_set(record, op):
    v_org = IF.get_mrc(record['subtomogram'])
    v = (v_org - v_org.mean())
    if not op['density_positive']:
        v = (- v)
    if 'smooth' in op:
        vg = FG.smooth(v, sigma=op['smooth']['sigma'])
    else:
        vg = v
    phi = N.sign((vg - N.abs(vg).mean()))
    phi = SA.segment(vg, phi, op['ac_segmentation']['smooth_weight'],
                     (op['ac_segmentation']['image_weight'] / vg.var()),
                     print_progress=op['ac_segmentation']['print_progress'])
    if phi is None:
        return
    if (phi > 0).sum() == 0:
        return
    if (phi < 0).sum() == 0:
        return
    if vg[(phi > 0)].mean() < vg[(phi < 0)].mean():
        phi = (- phi)
    return {'phi': phi, 'v_org': v_org, 'v': v, 'vg': vg, }


def normalize(record, op):
    if os.path.isfile(record['pose']['subtomogram']):
        return {'record': record, }
    ls = level_set(record=record, op=op['segmentation'])
    if ls is None:
        return
    phi = N.zeros(ls['phi'].shape)
    phi[(ls['phi'] > 0)] = ls['phi'][(ls['phi'] > 0)]
    c = PNU.center_mass(phi)
    mid_co = (N.array(phi.shape) / 2)
    if N.sqrt(N.square((c - mid_co)).sum()) > (N.min(phi.shape) * op['center_mass_max_displacement_proportion']):
        return
    rm = PNU.pca(v=phi, c=c)['v']
    record['pose']['c'] = c.tolist()
    record['pose']['rm'] = rm.tolist()
    phi_pn = GR.rotate(phi, rm=rm, c1=c, default_val=0)
    v_org_pn = GR.rotate_pad_mean(ls['v_org'], rm=rm, c1=c)
    return {'ls': ls, 'phi': phi, 'phi_pn': phi_pn, 'v_org_pn': v_org_pn, 'record': record, }


def main(op, pool=None, n_chunk=1000):
    op['data_config out'] = os.path.abspath(op['data_config out'])
    if not os.path.isdir(os.path.dirname(op['data_config out'])):
        os.makedirs(os.path.dirname(op['data_config out']))
    with open(op['data_config in']) as f:
        data = json.load(f)
    for d in data:
        if not os.path.isabs(d['subtomogram']):
            d['subtomogram'] = os.path.abspath(os.path.join(os.path.dirname(op['data_config in']), d['subtomogram']))
        if not os.path.isabs(d['mask']):
            d['mask'] = os.path.abspath(os.path.join(os.path.dirname(op['data_config in']), d['mask']))
    op['common_path'] = os.path.commonprefix([_['subtomogram'] for _ in data])
    if os.path.isfile(op['data_config out']):
        with open(op['data_config out']) as f:
            data_new = json.load(f)
        subtomograms_processed = set([_['subtomogram'] for _ in data_new])
        print('loaded', len(subtomograms_processed), 'processed subtomograms')
    else:
        data_new = []
        subtomograms_processed = set()
    if 'multiprocessing' not in op:
        op['multiprocessing'] = False
    if op['multiprocessing']:
        if pool is None:
            pool = Pool()
        pool_apply = []
        for (i, r) in enumerate(data):
            if r['subtomogram'] in subtomograms_processed:
                continue
            out_path_root = os.path.join(os.path.abspath(op['out dir']), r['subtomogram'][len(op['common_path']):])
            out_path_root = os.path.splitext(out_path_root)[0]
            assert ('segmentation' not in r)
            r['pose'] = {}
            r['pose']['subtomogram'] = (out_path_root + '-seg-pn.mrc')
            r['i'] = i
            pool_apply.append(pool.apply_async(func=do_normalize, kwds={'record': r, 'op': op, }))
        for pa in pool_apply:
            r = pa.get(99999999)
            if r is None:
                continue
            print('\rprocessing subtomogram', r['record']['i'], '             ', 'successfully processed',
                  len(data_new), '                             ', end=' ')
            sys.stdout.flush()
            del r['record']['i']
            if False:
                rs = save_data(r, op)
                data_new.append(rs)
            else:
                data_new.append(r['record'])
            if (len(data_new) % n_chunk) == 0:
                with open(op['data_config out'], 'w') as f:
                    json.dump(data_new, f, indent=2)
        del pool
        del pool_apply
    else:
        for (i, r) in enumerate(data):
            if r['subtomogram'] in subtomograms_processed:
                continue
            out_path_root = os.path.join(os.path.abspath(op['out dir']), r['subtomogram'][len(op['common_path']):])
            out_path_root = os.path.splitext(out_path_root)[0]
            assert ('segmentation' not in r)
            r['pose'] = {}
            r['pose']['subtomogram'] = (out_path_root + '-seg-pn.mrc')
            nr = normalize(record=r, op=op)
            if nr is None:
                continue
            data_new.append(nr['record'])
            print('\rprocessing subtomogram', i, '                    ', 'successfully processed', len(data_new),
                  '                             ', end=' ')
            sys.stdout.flush()
            if (len(data_new) % n_chunk) == 0:
                with open(op['data_config out'], 'w') as f:
                    json.dump(data_new, f, indent=2)
    print('successfully pose normalized subtomograms in totoal:', len(data_new))
    with open(op['data_config out'], 'w') as f:
        json.dump(data_new, f, indent=2)
