"""
Computes a list of hypervolumes of the voronoi regions given
a list of points in n dimensional space and a list of boundaries
"""

import aitom.geometry.ang_loc as TGA
import numpy as np
# import tomominer.hypervolume.hyper as hyper


def voronoi_hypervolumes(points, boundaries=None):
    n = len(points)
    d = len(points[0])
    if boundaries is None:
        # Initialize the boundaries
        # so that the minimum and maximum along each dimension
        # is determined by the minimum and maximum values amoung the points
        boundaries = [{'type': 'clip'} for _ in range(d)]

    if len(boundaries) != d:
        print("Error in voronoi_hypervolumes: dimension mismatch!\n")
        exit(1)

    prefix = [None, d + 1.0]

    wrap = [False for _ in range(d)]
    for i in range(d):
        if boundaries[i]['type'] == 'wrap':
            wrap[i] = True
        else:
            if boundaries[i]['type'] == 'range':
                lo, hi = boundaries[i]['range']
            else:
                i_proj = [p[i] for p in points]
                lo, hi = min(i_proj), max(i_proj)
            A = [0.0] * i + [1.0] + [0.0] * (d - i - 1)
            prefix += [-lo] + A
            A = [0.0] * i + [-1.0] + [0.0] * (d - i - 1)
            prefix += [hi] + A

    result = [None for _ in range(n)]

    for i in range(n):
        info = prefix
        for k in range(d):
            if wrap[k]:
                lo, hi = points[i][k] - np.pi, points[i][k] + np.pi
                A = [0.0] * k + [1.0] + [0.0] * (d - k - 1)
                info = info + [-lo] + A
                A = [0.0] * k + [-1.0] + [0.0] * (d - k - 1)
                info = info + [hi] + A
        for j in range(n):
            if i != j:
                info = info + get_hyperplanes(points[i], points[j], wrap)
        info[0] = (len(info) - 2) / (d + 1)
        print("Number of planes: %d" % info[0])
        result[i] = hyper.compute_hypervolume(np.asarray(info))

    return result


def get_hyperplanes(p1, p2, wrap):
    d = len(wrap)
    p2_list = [p2]
    wrap_amount = 2.0 * np.pi
    for i in range(d):
        if wrap[i]:
            list_lo = []
            list_hi = []
            for p_ in p2_list:
                lo = np.copy(p_)
                lo[i] -= wrap_amount
                list_lo.append(lo)
                hi = np.copy(p_)
                hi[i] += wrap_amount
                list_hi.append(hi)
            p2_list += list_lo + list_hi
    result = []
    for p_ in p2_list:
        result = result + get_hyperplane(p1, p_)
    return result


def get_hyperplane(p1, p2):
    vp = p2 - p1
    b = np.dot(vp, vp) / 2.0 + np.dot(p1, vp)
    result = [b] + (-vp).tolist()
    return result


def voronoi_weights_6d(phis):
    num_samples = 10000
    n = len(phis)
    result = np.zeros(n, dtype='float32')
    xl = [phi['q_x'] for phi in phis]
    yl = [phi['q_y'] for phi in phis]
    zl = [phi['q_z'] for phi in phis]
    sample_range = [[min(xl), max(xl)],
                    [min(yl), max(yl)],
                    [min(zl), max(zl)],
                    [0, np.pi * 2],
                    [0, np.pi * 2],
                    [0, np.pi * 2]]
    points = []
    for phi in phis:
        p = [phi['q_x'], phi['q_y'], phi['q_z'],
             phi['q_rot'], phi['q_tilt'], phi['q_psi']]
        p = np.asarray(p)
        points.append(p)

    for _ in range(num_samples):
        p_ = []
        for r in sample_range:
            p_.append(np.random.uniform(r[0], r[1]))

    best_d = None
    best_i = None
    for i in range(n):
        p = points[i]
        d = distance_6d_sq__frobenius(p, p_)
        if best_d is None or d < best_d:
            best_i = i
            best_d = d
    result[best_i] += 1
    result = result / np.sum(result)
    # print (result)
    return result


def distance_6d_sq__frobenius(p1, p2, weight=1.0):
    """
    square of the distance defined on the manifold of the 6D rigid transformation parameter space, the distance between
    rotations are calculated using Frobenius norm
    """
    loc1 = np.array(p1[:3])
    assert len(loc1) == 3
    ang1 = p1[3:]
    assert len(ang1) == 3
    rm1 = TGA.rotation_matrix_zyz(ang1)

    loc2 = np.array(p2[:3])
    assert len(loc2) == 3
    ang2 = p2[3:]
    assert len(ang2) == 3
    rm2 = TGA.rotation_matrix_zyz(ang2)

    d_rm = np.square(np.eye(3) - rm1.transpose().dot(rm2)).sum()
    d_loc = np.square(loc1 - loc2).sum()

    return d_rm + weight * d_loc


def voronoi_weights_6d_rlass(phis):
    points = []
    boundaries = [{'type': 'clip'}, {'type': 'clip'}, {'type': 'clip'},
                  {'type': 'clip'}, {'type': 'clip'}, {'type': 'clip'}]
    for phi in phis:
        p = [phi['q_x'], phi['q_y'], phi['q_z'],
             phi['q_rot'], phi['q_tilt'], phi['q_psi']]
        p = np.asarray(p)
        points.append(p)

    volumes = voronoi_hypervolumes(points, boundaries)
    volumes = np.asarray(volumes)
    result = volumes / (np.sum(volumes))
    print(result)
    return result


def test():
    tests = [
        [
            [(0, 0)],
            [(-1, 1), (-1, 1)]
        ],
        [
            [(-1, 0), (1, 0)],
            [(-1, 1), (-1, 1)]
        ],
        [
            [(1, 0), (0, 1), (-1, 0), (0, -1)],
            [(-1, 1), (-1, 1)]
        ],
        [
            [(1, 1), (-1, 1), (1, -1), (-1, -1)],
            [(-1, 1), (-1, 1)]
        ],
        [
            [(1, 1), (-1, 1), (-1, -1)],
            [(-1, 1), (-1, 1)]
        ],
        [
            [(1, 0), (-1, 1), (1, -1), (-1, -1)],
            [(-1, 1), (-1, 2)]
        ],
        [
            [(1, 1), (-1, 1), (0, 1), (-1, -1)],
            [(-1, 1), (-1, 1)]
        ],
        [
            [(1, 1, 0), (-1, 1, 0), (0, 1, 0), (-1, -1, 0)],
            [(-1, 1), (-1, 1), (-1, 1)]
        ],
        [
            [(1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1)],
            [(-1, 1), (-1, 1), (-1, 1)]
        ]
    ]
    for t in tests:
        points = [np.asarray(p) for p in t[0]]
        boundaries = [{'type': 'range', 'range': r} for r in t[1]]
        print("Points: ", t[0])
        print("Boundaries: ", t[1])
        print("Results: ", voronoi_hypervolumes(points, boundaries))
    # Test cliping
    tests = [[(1, 0), (0, 1), (-1, 0), (0, -1)],
             [(1, 1), (-1, 1), (1, -1), (-1, -1)],
             [(1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1)]
             ]
    for t in tests:
        points = [np.asarray(p) for p in t]
        print("Points: ", t)
        print("Results: ", voronoi_hypervolumes(points))
    # Test wrapping
    tests = [  # [(0,)],
        [(0,), (1,)],
        [(0,), (1,), (2,)],
        [(-1, -1), (1, 1)],
        [(1, 0), (0, 1), (-1, 0), (0, -1)],
        [(1, 1), (-1, 1), (1, -1), (-1, -1)],
        [(1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1)]
    ]
    for t in tests:
        points = [np.asarray(p, dtype='float32') for p in t]
        boundaries = [{'type': 'wrap'} for _ in t[0]]
        print("Points: ", t)
        print("Results: ", voronoi_hypervolumes(points, boundaries))
