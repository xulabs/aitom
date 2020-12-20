"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import os, sys, copy, uuid, time
import numpy as N
import aitom.tomominer.statistics.ssnr as SS
import socket


def data_prepare(dj, op):
    print('data_prepare()')
    if 'segmentation_tg_op' not in op:
        op['segmentation_tg_op'] = None
    s_sum = None
    s_prod_sum = None
    s_mask_sum = None
    for (i, d) in enumerate(dj):
        print(i, '      ', '\r', end=' ')
        sys.stdout.flush()
        r = SS.var__local(self=None, data_json=[d], return_key=False, segmentation_tg_op=op['segmentation_tg_op'])
        if s_sum is None:
            siz = [len(dj)]
            siz.extend(r['sum'][0].shape)
            s_sum = N.zeros(siz, dtype=N.complex)
            s_prod_sum = N.zeros(siz, dtype=N.complex)
            s_mask_sum = N.zeros(siz)
        s_sum[i, :, :, :] = r['sum'][0]
        s_prod_sum[i, :, :, :] = r['prod_sum'][0]
        s_mask_sum[i, :, :, :] = r['mask_sum'][0]
    return {'sum': s_sum, 'prod_sum': s_prod_sum, 'mask_sum': s_mask_sum, }


def ga(self=None, stat=None, initial_population=None, op=None):
    c_len = len(stat['sum'])
    n = op['population_size']
    if initial_population is not None:
        p = ga__init(p0=initial_population, size=n)
    else:
        assert (op['init']['rand_threshold'] > 0)
        assert (op['init']['rand_threshold'] < 1)
        p = (N.random.random((op['population_size'], c_len)) >= op['init']['rand_threshold']).astype(N.int)
    if 'parallel' in op:
        if 'host' not in op['parallel']['redis']:
            op['parallel']['redis']['host'] = socket.gethostname()
        stat_keys_key = ga__parallel__stat_distribute(self, stat, batch_id=str(uuid.uuid4()), op=op['parallel'])
    best_score = None
    best = None
    for iter_i in range(op['max_iteration_num']):
        if 'parallel' in op:
            gep_op = copy.deepcopy(op['evaluate'])
            gep_op['parallel'] = copy.deepcopy(op['parallel'])
            e = ga_evaluate__parallel(self=self, p=p, stat_keys_key=stat_keys_key, op=gep_op)
        else:
            e = ga_evaluate(p=p, stat=stat, op=op['evaluate'])
        combined = ga__combine_population_ordered(c0=best, c1={'p': copy.deepcopy(p), 'e': copy.deepcopy(e), })
        p = ga_evolve(p0=combined['p'], s=N.array([_['score'] for _ in combined['e']]), mr=op['mutation_rate'], n=n,
                      sum_min=op['sum_min'])
        best = {'p': combined['p'][:n, ], 'e': combined['e'][:n], }
        max_i = N.argmax([_['score'] for _ in combined['e']])
        best_score_t = combined['e'][max_i]['score']
        if (best_score is None) or (best_score_t > best_score['score']):
            best_score = {'score': best_score_t, 'repeat': 0, }
        else:
            best_score['repeat'] += 1
        sel_nums = [N.sum(_) for _ in combined['p']]
        print('\r', '\t\t', ('iter_i %4d' % (iter_i,)), ('score %3.4f' % (best_score_t,)), '        ',
              ('best_rep %3d' % (best_score['repeat'],)), '        ',
              ('subtomogram num %6d' % (int(N.sum(combined['p'][max_i, :])),)), '        ', 'min', N.min(sel_nums),
              'max', N.max(sel_nums), end=' ')
        sys.stdout.flush()
        if best_score['repeat'] >= op['best_score_max_repeat']:
            break
    if 'parallel' in op:
        ga__parallel__stat_cleanup(self=self, stat_keys_key=stat_keys_key)
    print()
    return best


def ga_evaluate__single(l, stat, op):
    l = l.astype(bool)
    sum_v = N.sum(stat['sum'][l, :, :, :], axis=0)
    prod_sum = N.sum(stat['prod_sum'][l, :, :, :], axis=0)
    mask_sum = N.sum(stat['mask_sum'][l, :, :, :], axis=0)
    st = SS.ssnr__given_stat(sum_v=sum_v, prod_sum=prod_sum, mask_sum=mask_sum, op=op['ssnr'])
    st['ssnr_log_sum'] = N.sum(N.log(st['fsc']))
    st['fsc_sum'] = N.sum(st['fsc'])
    return st


def ga_evaluate__scoring(s, op):
    if op['method'] == 'fsc_sum':
        for ss in s:
            ss['score'] = ss['fsc_sum']
    elif op['method'] == 'ssnr_log_sum':
        for ss in s:
            ss['score'] = ss['ssnr_log_sum']
    else:
        raise Exception('method')


def ga_evaluate(p, stat, op):
    s = ([None] * len(p))
    for (i, l) in enumerate(p):
        s[i] = ga_evaluate__single(l=l, stat=stat, op=op)
    ga_evaluate__scoring(s, op=op['scoring'])
    return s


def ga__combine_population_ordered(c0, c1):
    p = None
    e = None
    if c0 is not None:
        p = c0['p']
        e = c0['e']
        if c1 is not None:
            p = N.vstack((p, c1['p']))
            e.extend(c1['e'])
    elif c1 is not None:
        p = c1['p']
        e = c1['e']
    s = N.array([_['score'] for _ in e])
    i = N.argsort((- s))
    p = p[i, :]
    e = [e[_] for _ in i]
    return {'p': p, 'e': e, }


def ga_evolve(p0, s, mr, n=None, sum_min=None):
    assert (sum_min is not None)
    cdf = ga_evolve__make_ecdf(s)
    if n is None:
        n = p0.shape[0]
    p = (N.zeros((n, p0.shape[1])) + N.nan)
    c = 0
    while c < n:
        (i0, i1) = ga_evolve__select_pair(c=cdf)
        (l0, l1) = ga_evolve__crossover(p0[i0], p0[i1])
        pt = ga_evolve__mutate(l=l0, mr=mr)
        if pt.sum() < sum_min:
            continue
        p[c, :] = pt
        c += 1
        if c >= n:
            break
        pt = ga_evolve__mutate(l=l1, mr=mr)
        if pt.sum() < sum_min:
            continue
        p[c, :] = pt
        c += 1
    return p.astype(N.int)


def ga_evolve__make_ecdf(s):
    s = (s - N.min(s))
    s_sum = N.sum(s)
    if s_sum > 0:
        s = (s / float(s_sum))
    else:
        assert N.isfinite(s_sum)
        s = (N.ones(len(s)) * (1.0 / len(s)))
    c = 0
    cdf = N.zeros(len(s))
    for i in range(len(s)):
        c += s[i]
        cdf[i] = c
    return cdf


def ga_evolve__sample_ecdf(c):
    return N.sum((c < N.random.uniform()))


def ga_evolve__select_pair(c, max_trial=1000):
    i0 = ga_evolve__sample_ecdf(c)
    c = 0
    while True:
        i1 = ga_evolve__sample_ecdf(c)
        if i0 != i1:
            break
        c += 1
        assert (c < max_trial)
    return i0, i1


def ga_evolve__crossover(c0, c1):
    m = len(c0)
    i = N.random.randint(m)
    l0 = (N.zeros(m, dtype=N.int) + N.nan)
    l1 = (N.zeros(m, dtype=N.int) + N.nan)
    l0[:i] = c0[:i]
    l0[i:] = c1[i:]
    l1[:i] = c1[:i]
    l1[i:] = c0[i:]
    return l0, l1


def ga_evolve__mutate(l, mr):
    l = N.copy(l)
    for i in range(len(l)):
        if N.random.uniform() > mr:
            continue
        if l[i] == 1:
            l[i] = 0
        elif l[i] == 0:
            l[i] = 1
        else:
            raise Exception()
    return l
