import numpy as N
import math


def rotation_matrix_zyz(ang):
    phi = ang[0]
    theta = ang[1]
    psi_t = ang[2]

    # first rotate about z axis for angle psi_t
    a1 = rotation_matrix_axis(2, psi_t)
    a2 = rotation_matrix_axis(1, theta)
    a3 = rotation_matrix_axis(2, phi)

    # for matrix left multiplication
    rm = a3.dot(a2).dot(a1)

    # note: transform because tformarray use right matrix multiplication
    rm = rm.transpose()

    return rm


def rotation_matrix_axis(dim, theta):
    """
    following are left handed system (clockwise rotation)
    IMPORTANT: different to MATLAB version, this dim starts from 0, instead of 1
    """
    # x-axis
    if dim == 0:
        rm = N.array(
            [[1.0, 0.0, 0.0], [0.0, math.cos(theta), -math.sin(theta)], [0.0, math.sin(theta), math.cos(theta)]])
    # y-axis
    elif dim == 1:
        rm = N.array(
            [[math.cos(theta), 0.0, math.sin(theta)], [0.0, 1.0, 0.0], [-math.sin(theta), 0.0, math.cos(theta)]])
    # z-axis
    elif dim == 2:
        rm = N.array(
            [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    else:
        raise

    return rm


def random_rotation_matrix():
    """generate a random rotation matrix using SVD on a random matrix"""
    m = N.random.random((3, 3))
    u, s, v = N.linalg.svd(m)

    return u


def random_rotation_angle_zyz():
    rm = random_rotation_matrix()
    return rotation_matrix_zyz_normalized_angle(rm)


def rotation_matrix_zyz_normalized_angle(rm):
    assert (all(N.isreal(rm.flatten())));
    assert (rm.shape == (3, 3));

    cos_theta = rm[2, 2]
    if N.abs(cos_theta) > 1.0:
        # warning(sprintf('cos_theta %g', cos_theta));
        cos_theta = N.sign(cos_theta)

    theta = N.arctan2(N.sqrt(1.0 - (cos_theta * cos_theta)), cos_theta)

    # use a small epslon to increase numerical stability when abs(cos_theta) is very close to 1!!!!
    if N.abs(cos_theta) < (1.0 - 1e-10):
        phi = N.arctan2(rm[2, 1], rm[2, 0])
        psi_t = N.arctan2(rm[1, 2], -rm[0, 2])
    else:
        theta = 0.0
        phi = 0.0
        psi_t = N.arctan2(rm[0, 1], rm[1, 1])

    ang = N.array([phi, theta, psi_t], dtype=N.float)

    return ang


def reverse_transform(rm, loc_r):
    rev_rm = rm.T
    rev_loc_r = (- N.dot(loc_r, rev_rm))
    return rev_rm, rev_loc_r


def reverse_transform_ang_loc(ang, loc_r):
    rm = rotation_matrix_zyz(ang)
    (rev_rm, rev_loc_r) = reverse_transform(rm, loc_r)
    return rotation_matrix_zyz_normalized_angle(rev_rm), rev_loc_r
