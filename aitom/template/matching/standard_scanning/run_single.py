"""
run a single job, of calculating correlation for a angle centain range for a particular partition
~/ln/tomominer/tomominer/template/search/standard_scanning/batch/run_single.py
"""

import os
import sys
import json
import time
import cPickle as pickle
import numpy as N
import scipy.ndimage.filters as SNF

import aitom.io.file as IOF
import aitom.geometry.rotate as GR
import aitom.filter.convolve as FC
import aitom.filter.normalized_cross_correlation as FNCC
import aitom.filter.local_extrema as FLE


def scan(op):
    if not os.path.isdir(op['out_dir']):
        os.makedirs(op['out_dir'])

    re = {'c': os.path.join(op['out_dir'], '%s-c.npy' % (op['id'],)),
          'phi': os.path.join(op['out_dir'], '%s-phi.npy' % (op['id'],)),
          'theta': os.path.join(op['out_dir'], '%s-theta.npy' % (op['id'],)),
          'psi': os.path.join(op['out_dir'], '%s-psi.npy' % (op['id'],))}

    if os.path.isfile(re['c']) and os.path.isfile(re['phi']) and os.path.isfile(re['theta']) and os.path.isfile(
        re['psi']):
        return re

    t = N.load(op['template'])
    # real space mask
    tm = N.isfinite(t)
    t_mean = t[tm].mean()
    t[N.logical_not(tm)] = t_mean
    tm = tm.astype(N.float)

    v = N.load(op['map'])

    print('map size', v.shape)
    sys.stdout.flush()

    diff_time_v = []
    cur_time = time.time()

    c_max = None
    phi_max = None
    theta_max = None
    psi_max = None

    for i, (phi, theta, psi) in enumerate(op['angles']):
        tr = GR.rotate(t, angle=(phi, theta, psi), default_val=t_mean)

        if op['mode'] == 'convolve':
            # c = FC.convolve(v=v, t=tr)
            c = SNF.convolve(input=v, weights=tr, mode='reflect')
        elif op['mode'] == 'normalized-cor':
            tmr = GR.rotate(tm, angle=(phi, theta, psi), default_val=0.0)
            tr[tmr < 0.5] = float('NaN')
            c = FNCC.cor(v=v, t=tr)
        else:
            raise Exception('mode')

        if c_max is None:
            c_max = c
            phi_max = N.zeros(c.shape) + phi
            theta_max = N.zeros(c.shape) + theta
            psi_max = N.zeros(c.shape) + psi

        else:

            ind = (c > c_max)
            c_max[ind] = c[ind]
            phi_max[ind] = phi
            theta_max[ind] = theta
            psi_max[ind] = psi

        diff_time = time.time() - cur_time
        diff_time_v.append(diff_time)
        remain_time = N.array(diff_time_v).mean() * (len(op['angles']) - i - 1)
        print('angle', i, 'of', len(op['angles']), '; time used', diff_time, '; time remain', remain_time)
        sys.stdout.flush()

        cur_time = time.time()

    if not os.path.isfile(re['c']):
        N.save(re['c'], c_max)
        N.save(re['phi'], phi_max)
        N.save(re['theta'], theta_max)
        N.save(re['psi'], psi_max)

    return re


def main():
    job_file = os.environ['job_file']
    assert job_file is not None
    print('loading job from', job_file)

    with open(job_file, 'rb') as f:
        jop = pickle.load(f)
    with open(jop['config_file']) as f:
        cop = json.load(f)
    with open(jop['config_stat_out_file']) as f:
        cop_out = json.load(f)

    print('to process', len(jop['job']['angles']), 'rotation angles')
    sys.stdout.flush()

    sop = {'id': jop['id'],
           'out_dir': jop['out_dir'],
           'template': cop_out['template_tmp'],
           'mode': cop['mode'],
           'angles': jop['job']['angles'],
           'map': jop['job']['partition']['map_file']}

    s = scan(sop)

    with open(jop['stat_out'], 'w') as f:
        json.dump(s, f, indent=2)

    print('program finished')


if __name__ == "__main__":
    main()
