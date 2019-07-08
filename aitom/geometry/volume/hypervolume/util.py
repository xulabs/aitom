'''
Functions for hypervolume analysis
'''

def voronoi_weights_6d(phis):
    num_samples = 10000
    n = len(phis)
    result = np.zeros(n, dtype='float32')
    xl = [phi['q_x'] for phi in phis]
    yl = [phi['q_y'] for phi in phis]
    zl = [phi['q_z'] for phi in phis]
    sample_range = [[min(xl),max(xl)],
            [min(yl),max(yl)],
            [min(zl),max(zl)],
            [0,np.pi*2],
            [0,np.pi*2],
            [0,np.pi*2]
           ]
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
        if best_d == None or d < best_d:
            best_i = i
            best_d = d
    result[best_i] += 1
    result = result / np.sum(result)
    #print (result)
    return result


import tomominer_mbc.geometry.ang_loc as TGA
'''
square of the distance defined on the manifold of the 6D rigid transformation parameter space, the distance between rotations are calculated using Frobenius norm
'''
def distance_6d_sq__frobenius(p1, p2, weight=1.0):

    loc1 = np.array(p1[:3]);    assert len(loc1) == 3
    ang1 = p1[3:];      assert len(ang1) == 3
    rm1 = TGA.rotation_matrix_zyz(ang1)

    loc2 = np.array(p2[:3]);    assert len(loc2) == 3
    ang2 = p2[3:];      assert len(ang2) == 3
    rm2 = TGA.rotation_matrix_zyz(ang2)

    d_rm = np.square(np.eye(3) - rm1.transpose().dot(rm2)).sum()
    d_loc = np.square(loc1 - loc2).sum()

    return d_rm + weight*d_loc

