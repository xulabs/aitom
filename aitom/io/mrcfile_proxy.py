"""
access mrc files using mrcfile package
"""

import numpy as np
import mrcfile


def read(path):
    with mrcfile.open(path, 'r') as m:
        header = m.header
        data = m.data
        # only for 3D array
        assert data.ndim == 3
        # transpose data according to tomominer.image.vol.eman2_util.em2numpy
        data = data.transpose([2, 1, 0])

    return {'header': header, 'data': data}


def read_data(path):
    mrc = mrcfile.open(path, mode='r+', permissive=True)
    a = mrc.data
    assert a.shape[0] > 0
    a = a.astype(np.float32)
    a = a.transpose([2, 1, 0])

    return a


def read_header(path):
    from mrcfile.mrcinterpreter import MrcInterpreter

    mi = MrcInterpreter(iostream=open(path, 'rb'))
    mi._read_header()
    mi._iostream.close()

    return mi.header


def write_data(data, path, overwrite=False):
    # only for 3D array
    assert data.ndim == 3
    data = data.astype(np.float32)
    # transpose data according to tomominer.image.vol.eman2_util.em2numpy
    data = data.transpose([2, 1, 0])
    with mrcfile.new(path, overwrite=overwrite) as m:
        m.set_data(data)
