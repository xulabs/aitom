"""
utility functions for particle picking using Difference of Gaussian
"""

import os, sys, time, uuid, pickle
import gc as GC
import multiprocessing
from multiprocessing.pool import Pool as Pool
import numpy as N
import scipy.spatial.distance as SPD

import aitom.filter.gaussian as FG
import aitom.filter.local_extrema as FL
import aitom.io.file as IOF


def peak(v, s1, s2, find_maxima=True):
    """
    peak detection

    parameters:
        v: vol map
        s1: sigma1
        s2: sigma2

    return:
        x: coordinates
        p: peak values
        vg: dog smoothed vol map
        find_maxima: find local maxima or minima
    """
    p = peak__partition__single_job(v=v, s1=s1, s2=s2, find_maxima=find_maxima)

    return p['ps']


def peak__partition(v, s1, s2, find_maxima=True, partition_op=None, multiprocessing_process_num=0):
    """
    partition the volume then detect peaks for each partition
    note that this will result in redundant peaks!!
    Clean up must be done afterwards!!
    """
    import aitom.image.vol.partition as IVP

    if multiprocessing_process_num > 0:
        pool = Pool(processes=min(multiprocessing_process_num, multiprocessing.cpu_count()))
    else:
        pool = None

    if partition_op is None:
        # in this case, just generate a single partition
        siz_max = max(v.shape)
        partition_op = {'nonoverlap_width': siz_max * 2, 'overlap_width': siz_max * 2}

    b = IVP.gen_bases(v.shape, nonoverlap_width=partition_op['nonoverlap_width'],
                      overlap_width=partition_op['overlap_width'])
    print('partition num', b.shape)

    ps = []

    if pool is not None:
        pool_re = []
        for i0 in range(b.shape[0]):
            for i1 in range(b.shape[1]):
                for i2 in range(b.shape[2]):
                    bp = N.squeeze(b[i0, i1, i2, :, :])
                    pool_re.append(pool.apply_async(func=peak__partition__single_job, kwds={
                        'v': v[bp[0, 0]:bp[0, 1], bp[1, 0]:bp[1, 1], bp[2, 0]:bp[2, 1]],
                        's1': s1,
                        's2': s2,
                        'base': bp,
                        'find_maxima': find_maxima,
                        'partition_id': (i0, i1, i2),
                        'save_vg': (partition_op['save_vg'] if 'save_vg' in partition_op else False)}))

        for pool_re_t in pool_re:
            ppsj = pool_re_t.get(9999999)
            ps.extend(ppsj['ps'])
            print('\r', ppsj['partition_id'], '                     ')
            sys.stdout.flush()

        pool.close()
        pool.join()
        del pool

    else:

        for i0 in range(b.shape[0]):
            for i1 in range(b.shape[1]):
                for i2 in range(b.shape[2]):
                    bp = N.squeeze(b[i0, i1, i2, :, :])
                    ppsj = peak__partition__single_job(v=v[bp[0, 0]:bp[0, 1], bp[1, 0]:bp[1, 1], bp[2, 0]:bp[2, 1]],
                                                       s1=s1, s2=s2, base=bp, find_maxima=find_maxima,
                                                       partition_id=(i0, i1, i2), save_vg=(
                            partition_op['save_vg'] if 'save_vg' in partition_op else False))
                    ps.extend(ppsj['ps'])
                    print('\r', ppsj['partition_id'], '                     ')
                    sys.stdout.flush()

    # order peaks in ps according to values
    if find_maxima:
        ps = sorted(ps, key=lambda _: (-_['val']))
    else:
        ps = sorted(ps, key=lambda _: _['val'])

    return ps


def peak__partition__single_job(v, s1, s2, base=None, find_maxima=None, partition_id=None, save_vg=False):
    assert find_maxima is not None

    # vg = FG.dog_smooth__large_map(v, s1=s1, s2=s2) 'dog_smooth' seems to perform better
    vg = FG.dog_smooth(v, s1=s1, s2=s2)

    # save the smoothed partition for inspection
    if save_vg:
        IOF.put_mrc(v, '/tmp/%d-%d-%d--v.mrc' % (partition_id[0], partition_id[1], partition_id[2]), overwrite=True)

    del v

    if find_maxima:
        # print 'local_maxima()'
        # sys.stdout.flush()
        x = FL.local_maxima(vg)
    else:
        # print 'local_minima()'
        # sys.stdout.flush()
        x = FL.local_minima(vg)

    p = vg[x]
    x = N.array(x).T

    if base is not None:
        assert base.shape[0] == x.shape[1]
        assert base.shape[1] == 2

        for dim_i in range(x.shape[1]):
            x[:, dim_i] += base[dim_i, 0]

    ps = []
    for i in range(len(p)):
        ps.append({'val': float(p[i]), 'x': [_ for _ in x[i, :]], 'uuid': str(uuid.uuid4())})

    if save_vg:
        # save the smoothed partition for inspection
        IOF.put_mrc(vg, '/tmp/%d-%d-%d--vg.mrc' % (partition_id[0], partition_id[1], partition_id[2]), overwrite=True)

    del vg
    GC.collect()

    return {'ps': ps, 'partition_id': partition_id}
