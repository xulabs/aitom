'''
utilities for handling eman2 data
import tomominer.image.vol.eman2_util as TIVE
'''

import numpy as N
import EMAN2 as E

def numpy2em(v):
    assert v.ndim == 3      # only for 3D array
    vt = v.transpose([2,1,0]).copy()
    ve = E.EMNumPy.numpy2em(vt)
    return ve


def em2numpy(ve):
    assert ve.get_ndim() == 3           # only for 3D array
    return E.EMNumPy.em2numpy(ve).transpose([2,1,0]).copy()

'''
read mrc 3D image file into a numpy 3D array
'''
def read_mrc_numpy_vol(path):
    ve = E.EMData(str(path))
    v = em2numpy(ve)
    v = v.astype(N.float32)
    del ve      # to prevent memory leak
    return v


def read_mrc_numpy_vol__test():
    print 'read_mrc_numpy_vol__test()'
    import tomominer.model.util as TMU
    v = TMU.generate_toy_model()

    import tomominer.image.vol.util as TIVU
    v = TIVU.highlight_xy_axis(v)

    import tomominer.io.file as TIF
    TIF.put_mrc(v, '/tmp/v-tm.mrc')

    v_r = read_mrc_numpy_vol('/tmp/v-tm.mrc')
    print 'max difference', N.abs(v_r - v).max()

    import scipy.stats as SS
    print 'correlation', SS.pearsonr(v.flatten(), v_r.flatten())[0]





'''
write numpy 3D array into mrc 3D image file
'''
def write_numpy_vol_mrc(v, path):
    ve = numpy2em(v)
    wr = ve.write_image(str(path), 0, E.EMUtil.ImageType.IMAGE_MRC)
    del ve      # to prevent memory leak
    return wr


def write_numpy_vol_mrc__test():
    print 'write_numpy_vol_mrc__test()'

    import tomominer.model.util as TMU
    v = TMU.generate_toy_model()

    import tomominer.image.vol.util as TIVU
    v = TIVU.highlight_xy_axis(v)

    import tomominer.io.file as TIF
    TIF.put_mrc(v, '/tmp/v-tm.mrc')

    write_numpy_vol_mrc(v, '/tmp/v-em.mrc')

    v_r = TIF.read_mrc_data('/tmp/v-tm.mrc')
    ve_r = TIF.read_mrc_data('/tmp/v-em.mrc')

    print 'max difference', N.abs(v_r - ve_r).max()

    #ve.write_image('/tmp/v-em.hdf', v.shape[2])

    #vev = E.EMNumPy.em2numpy(ve).copy()
    #TIF.put_mrc(vev, '/tmp/v-em-tm.mrc')


'''
write numpy 3D array into MAP image file
'''
def write_numpy_vol_map(v, path):
    # this function is not tested yet
    ve = numpy2em(v)
    wr = ve.write_image(str(path), 0, E.EMUtil.ImageType.IMAGE_ICOS)
    del ve      # to prevent memory leak
    return wr


'''
write numpy 3D array into hdf 3D image file
'''
def write_numpy_vol_hdf(v, path):
    #return numpy2em(v).write_image(str(path), 0, E.EMUtil.ImageType.IMAGE_HDF)
    return numpy2em(v).write_image(str(path))


'''
read mrc 3D image file into a numpy 3D array
'''
def read_hdf_numpy_vol(path):
    ve = E.EMData(str(path))
    return em2numpy(ve)


def read_write_hdf_files__test():
    print 'read_write_hdf_files__test()'

    import tomominer.model.util as TMU
    v = TMU.generate_toy_model()

    import tomominer.image.vol.util as TIVU
    v = TIVU.highlight_xy_axis(v)

    import tomominer.io.file as TIF
    TIF.put_mrc(v, '/tmp/v-tm.mrc')

    write_numpy_vol_hdf(v, '/tmp/v-hdf.hdf')


