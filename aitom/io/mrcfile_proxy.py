'''
access mrc files using mrcfile package

'''

import numpy as N
import mrcfile

def read(path):
    with mrcfile.open(path, 'r') as m:
        header = m.header
        data = m.data
        assert data.ndim == 3  # only for 3D array
        data = data.transpose([2, 1, 0])        # this is according to tomominer.image.vol.eman2_util.em2numpy

    return {'header':header, 'data': data}

def read_data(path):

    mrc = mrcfile.open(path,mode='r+',permissive=True)
    a = mrc.data
    assert a.shape[0] > 0
    a = a.astype(N.float32) 
    a = a.transpose([2,1,0])
    
    return a

def read_header(path):
    from mrcfile.mrcinterpreter import MrcInterpreter

    mi = MrcInterpreter()
    mi._iostream = open(path, 'rb')
    mi._read_header()
    mi._iostream.close()

    return mi.header


def write_data(data, path, overwrite=False):
    assert data.ndim == 3  # only for 3D array

    data = data.astype(N.float32)
    data = data.transpose([2,1,0])        # this is according to tomominer.image.vol.eman2_util.numpy2em
    with mrcfile.new(path, overwrite=overwrite) as m:
        m.set_data(data)


