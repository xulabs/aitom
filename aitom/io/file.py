"""
Functions for reading and writing common files
"""

import pickle

from . import mrcfile_proxy as TIM
import aitom.tomominer.io.file as ATIF


def read_mrc_data(path, show_progress=False):
    return TIM.read_data(path)


def read_mrc_header(path):
    return ATIF.read_mrc(path=path, read_data=False)['header']


def get_mrc_voxel_spacing(im):
    """
    Calculate voxel spacing
    from the mrc header obtained from read_mrc_header
    in Angstrom unit
    """
    return im['header']['MRC']['xlen'] / im['header']['MRC']['nx']


def put_mrc(mrc, path, overwrite=False):
    put_mrc_data(mrc, path, overwrite=overwrite)


def put_mrc_data(mrc, path, overwrite=False):
    TIM.write_data(data=mrc, path=path, overwrite=overwrite)


def pickle_load(path):
    with open(path, 'rb') as f:
        o = pickle.load(f, encoding='iso-8859-1')
    return o


def pickle_dump(o, path, protocol=-1):
    with open(path, 'wb') as f:
        pickle.dump(o, f, protocol=protocol)
