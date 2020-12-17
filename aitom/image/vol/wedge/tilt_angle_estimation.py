#!/usr/bin/env python
"""
given a tomogram/subtomogram, guess the tilt angle range and light axis
"""

import gc
import itertools
import multiprocessing
import sys
import time

import numpy as N
from scipy.stats.stats import pearsonr

import aitom.model.util as MU
from .. import util as IVU


def wedge_mask_cor(v_abs, ops):
    for op in ops:
        m = tilt_mask(size=v_abs.shape, tilt_ang1=op['ang1'], tilt_ang2=op['ang2'], tilt_axis=op['tilt_axis'],
                      light_axis=op['light_axis'])
        # m = TIVWU.wedge_mask(size=v_abs.shape, ang1=op['ang1'], ang2=op['ang2'], tilt_axis=op['direction'])
        m = m.astype(N.float)

        op['cor'] = float(pearsonr(v_abs.flatten(), m.flatten())[0])

    return ops


def scan(tilt_angle_scan_range, v_abs, n_proc=0, n_chunk=20):
    from multiprocessing.pool import Pool
    pool = Pool(processes=n_proc)
    pool_results = []

    tasks = []
    for ang1, ang2, tilt_axis, light_axis in itertools.product(
            range(-tilt_angle_scan_range[1], -tilt_angle_scan_range[0] + 1),
            range(tilt_angle_scan_range[0], tilt_angle_scan_range[1] + 1), range(3), range(3)):
        if tilt_axis == light_axis:
            continue
        tasks.append({'ang1': ang1, 'ang2': ang2, 'tilt_axis': tilt_axis, 'light_axis': light_axis})

    while tasks:
        # wedge_mask_cor(v_abs=v, ops=tasks[:n_chunk])
        pool_results.append(pool.apply_async(func=wedge_mask_cor, kwds={'v_abs': v_abs, 'ops': tasks[:n_chunk]}))
        tasks = tasks[n_chunk:]

    best = None
    for re in pool_results:
        for r in re.get(9999999):
            # print this info so that we know the scanning is inside correct range, by looking at updated examples
            print('\r', r['ang1'], r['ang2'], r['tilt_axis'], r['light_axis'], r['cor'], '        ')
            sys.stdout.flush()

            if best is None:
                best = r
                continue

            if r['cor'] > best['cor']:
                best = r

    assert best is not None
    return best


def grid_distance_sq_to_center(grid):
    dist_sq = N.zeros(grid.shape[1:])
    if grid.ndim == 4:
        for dim in range(3):
            dist_sq += N.squeeze(grid[dim, :, :, :]) ** 2
    elif grid.ndim == 3:
        for dim in range(2):
            dist_sq += N.squeeze(grid[dim, :, :]) ** 2
    else:
        assert False

    return dist_sq


def grid_distance_to_center(grid):
    dist_sq = grid_distance_sq_to_center(grid)
    return N.sqrt(dist_sq)


def tilt_mask(size, tilt_ang1, tilt_ang2=None, tilt_axis=1, light_axis=2, sphere_mask_bool=True):
    assert tilt_axis != light_axis

    if tilt_ang2 is None:
        tilt_ang2 = float(N.abs(tilt_ang1))
        tilt_ang1 = -tilt_ang2

    else:
        assert tilt_ang1 < 0
        assert tilt_ang2 > 0

    tilt_ang1 = (tilt_ang1 / 180.0) * N.pi
    tilt_ang2 = (tilt_ang2 / 180.0) * N.pi

    g = IVU.grid_displacement_to_center(size=size, mid_co=IVU.fft_mid_co(siz=size))

    plane_axis = {0, 1, 2}
    plane_axis.difference_update([light_axis, tilt_axis])
    assert len(plane_axis) == 1
    plane_axis = list(plane_axis)[0]

    x_light = g[light_axis]
    x_plane = g[plane_axis]

    m = N.zeros(size, dtype=float)

    m[N.logical_and(x_light <= (N.tan(tilt_ang1) * x_plane), x_light >= (N.tan(tilt_ang2) * x_plane))] = 1.0
    m[N.logical_and(x_light >= (N.tan(tilt_ang1) * x_plane), x_light <= (N.tan(tilt_ang2) * x_plane))] = 1.0

    if sphere_mask_bool:
        m *= MU.sphere_mask(m.shape)

    return m


if __name__ == '__main__':
    # number of processes
    n_proc = 50
    n_proc = min(n_proc, multiprocessing.cpu_count())
    print('using', n_proc, 'cpus')

    from numpy.fft import fftshift, fftn
    import pickle

    # Download from: https://cmu.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp
    path = './aitom_demo_subtomograms.pickle'
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')

    # 'data' is a dict containing several different subtomograms.
    # 'data['5T2C_data']' is a list containing 100 three-dimensional arrays (100 subtomograms).
    print(data['5T2C_data'][0].shape)
    start_time = time.time()
    v = data['5T2C_data'][0]  # 32x32x32 volume
    v = v.astype(N.float)
    # it is very important to use log scale!!
    v = N.log(N.abs(fftshift(fftn(v))))
    # clean up memory before forking
    gc.collect()
    # scan range
    tilt_angle_scan_range = N.array([1, 89])
    tom_wedge = scan(tilt_angle_scan_range=tilt_angle_scan_range, v_abs=v, n_proc=n_proc)
    print('best tom_wedge: ', tom_wedge, time.time() - start_time, 's')
