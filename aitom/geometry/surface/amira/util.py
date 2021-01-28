"""
functions for parsing amira surf ascii file
"""

import copy
import numpy as N


def surf_parse(fo):
    vertice_num = None
    triangle_num = None

    # vertices
    v = []
    # triangles,  IMPORTANT: the indicis in triangels starts from 1 not 0!!!
    t = []

    import time
    while True:
        line = fo.readline()
        if not line:
            break
        line = line.strip()

        if line.startswith('Vertices '):
            assert vertice_num is None
            ss = line.split(' ')
            vertice_num = int(ss[1])

            for i in range(vertice_num):
                line = fo.readline()
                assert line
                line = line.strip()
                ss = line.split(' ')
                assert len(ss) == 3
                v.append([float(_) for _ in ss])

        if line.startswith('Triangles '):
            assert vertice_num is not None
            # for simplificity, current we only process one set of triangles
            # assert triangle_num is None
            ss = line.split(' ')
            triangle_num = int(ss[1])

            for i in range(triangle_num):
                line = fo.readline()
                assert line
                line = line.strip()
                ss = line.split(' ')
                assert len(ss) == 3
                t.append([int(_) for _ in ss])

    return {'vertices': N.array(v, dtype=N.float), 'faces': N.array(t, dtype=N.int)}


def export_surf_ascii_simple(s, f):
    """
    written according to Amira.export_surf_ascii_simple()
    parameters:
        s: surface structure
        f: file object
    """
    print('# HyperSurface 0.1 ASCII ')
    print('Vertices %d' % (s['vertices'].shape[0],))

    for i in range(s['vertices'].shape[0]):
        x = s['vertices'][i, :].flatten().tolist()
        print('%f %f %f ' % (x[0], x[1], x[2]))

    print('Patches 1')
    print('{ ')

    print('InnerRegion Inside ')
    print('Triangles %d ' % (s['faces'].shape[0],))

    faces = copy.deepcopy(s['faces'])
    if faces.min() < 1:
        # IMPORTANT: for amira, the index of faces must starts from at least 1!!
        faces -= faces.min()
        faces += 1

    for i in range(faces.shape[0]):
        j = faces[i, :].flatten().tolist()
        print('%d %d %d ' % (j[0], j[1], j[2]))

    print('} ')


def vertice_location_transform(x, mrc):
    """
    transform the vertex location info from voxel grid space to original tomogram space defined in MRC
    parameters:
        x: location
        mrc: MRC header
    """
    xt = N.zeros(x.shape, dtype=N.float)

    xt[:, 0] = (x[:, 0] / mrc['nx']) * mrc['xlen'] + mrc['xorg']
    xt[:, 1] = (x[:, 1] / mrc['ny']) * mrc['ylen'] + mrc['yorg']
    xt[:, 2] = (x[:, 2] / mrc['nz']) * mrc['zlen'] + mrc['zorg']

    return xt


def vertice_location_transform__amira_to_vol(x, mrc):
    """
    transform the vertex location info from original tomogram space defined in MRC to voxel grid space
    parameters:
        x: location
        mrc: MRC header
    """
    xt = N.zeros(x.shape, dtype=N.float)

    xt[:, 0] = mrc['nx'] * (x[:, 0] - mrc['xorg']) / mrc['xlen']
    xt[:, 1] = mrc['ny'] * (x[:, 1] - mrc['yorg']) / mrc['ylen']
    xt[:, 2] = mrc['nz'] * (x[:, 2] - mrc['zorg']) / mrc['zlen']

    return xt


if __name__ == "__main__":
    with open('/tmp/t.surf') as f:
        s = surf_parse(f)
        res = s['vertices'].shape[s['faces'].min(), s['faces'].max()]