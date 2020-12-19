"""
functions for calculating eigen values and eigen vectors
"""

import numpy as N



def eigen_value_3_symmetric_batch(A):
    """
    given a batch of 3 by 3 symmetric matrixes, calculate eigenvalues for each matrix.
    See paper Eberly06 Eigensystems for 3 by 3 Symmetric Matrices
    IMPORTANT: A is preconditioned by dividing by its maximum magnitude entry when that maximum is larger than 1
    """
    for At in A:
        for Att in At:
            if Att is None:
                continue
            assert Att.max() <= 1.0

    inv3 = 1.0 / 3.0
    root3 = float(N.sqrt(3.0))

    a00 = A[0][0].flatten()
    a01 = A[0][1].flatten()
    a02 = A[0][2].flatten()
    a11 = A[1][1].flatten()
    a12 = A[1][2].flatten()
    a22 = A[2][2].flatten()

    c0 = a00 * a11 * a22 + 2.0 * a01 * a02 * a12 - a00 * a12 * a12 - a11 * a02 * a02 - a22 * a01 * a01
    c1 = a00 * a11 - a01 * a01 + a00 * a22 - a02 * a02 + a11 * a22 - a12 * a12
    c2 = a00 + a11 + a22

    c2Div3 = c2 * inv3
    aDiv3 = (c1 - c2 * c2Div3) * inv3

    aDiv3[aDiv3 > 0.0] = 0.0

    mbDiv2 = 0.5 * (c0 + c2Div3 * (2.0 * c2Div3 * c2Div3 - c1))

    q = mbDiv2 * mbDiv2 + aDiv3 * aDiv3 * aDiv3

    q[q > 0.0] = 0.0

    magnitude = N.sqrt(-aDiv3)
    angle = N.arctan2(N.sqrt(-q), mbDiv2) * inv3

    cs = N.cos(angle)
    sn = N.sin(angle)

    root = N.zeros((a00.size, 3))
    root[:, 0] = c2Div3 + 2.0 * magnitude * cs
    root[:, 1] = c2Div3 - magnitude * (cs + root3 * sn)
    root[:, 2] = c2Div3 - magnitude * (cs - root3 * sn)

    # Sort the roots here to obtain abs(root[0]) >= (root[1]) >= (root[2]).
    root_i = N.argsort(-N.abs(root), axis=1)
    root = N.array([root[_, root_i[_]] for _ in range(root.shape[0])])

    re = [None] * 3
    for dim in range(3):
        re[dim] = N.reshape(root[:, dim], A[0][0].shape)

    return re
