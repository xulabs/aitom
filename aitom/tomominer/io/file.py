

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import os
import sys
import pickle
import shutil
import time
import struct
import uuid
import array
import numpy as N

def get_mrc(path, retry_interval=1.0, max_retry=5):
    path = os.path.realpath(str(path))
    import aitom.tomominer.core.core as tomo
    v = None
    retry = 0
    while (retry < max_retry):
        try:
            v = None
            v = tomo.read_mrc(path)
            break
        except:
            retry += 1
            time.sleep(retry_interval)
    if (v is None):
        raise IOError(('cannot load ' + path))
    return v

def read_mrc(path, read_data=True, show_progress=False):
    path = os.path.realpath(path)
    with open(path, 'rb') as f:
        mrc = {}
        mrc['nx'] = int(struct.unpack('i', f.read(4))[0])
        mrc['ny'] = int(struct.unpack('i', f.read(4))[0])
        mrc['nz'] = int(struct.unpack('i', f.read(4))[0])
        mrc['mode'] = struct.unpack('i', f.read(4))[0]
        mrc['nxstart'] = struct.unpack('i', f.read(4))[0]
        mrc['nystart'] = struct.unpack('i', f.read(4))[0]
        mrc['nzstart'] = struct.unpack('i', f.read(4))[0]
        mrc['mx'] = struct.unpack('i', f.read(4))[0]
        mrc['my'] = struct.unpack('i', f.read(4))[0]
        mrc['mz'] = struct.unpack('i', f.read(4))[0]
        mrc['xlen'] = struct.unpack('f', f.read(4))[0]
        mrc['ylen'] = struct.unpack('f', f.read(4))[0]
        mrc['zlen'] = struct.unpack('f', f.read(4))[0]
        mrc['alpha'] = struct.unpack('f', f.read(4))[0]
        mrc['beta'] = struct.unpack('f', f.read(4))[0]
        mrc['gamma'] = struct.unpack('f', f.read(4))[0]
        mrc['mapc'] = struct.unpack('i', f.read(4))[0]
        mrc['mapr'] = struct.unpack('i', f.read(4))[0]
        mrc['maps'] = struct.unpack('i', f.read(4))[0]
        mrc['amin'] = struct.unpack('f', f.read(4))[0]
        mrc['amax'] = struct.unpack('f', f.read(4))[0]
        mrc['amean'] = struct.unpack('f', f.read(4))[0]
        mrc['ispg'] = struct.unpack('h', f.read(2))[0]
        mrc['nsymbt'] = struct.unpack('h', f.read(2))[0]
        mrc['next'] = struct.unpack('i', f.read(4))[0]
        mrc['creatid'] = struct.unpack('h', f.read(2))[0]
        mrc['unused1'] = struct.unpack(('c' * 30), f.read(30))[0]
        mrc['nint'] = struct.unpack('h', f.read(2))[0]
        mrc['nreal'] = struct.unpack('h', f.read(2))[0]
        mrc['unused2'] = struct.unpack(('c' * 28), f.read(28))[0]
        mrc['idtype'] = struct.unpack('h', f.read(2))[0]
        mrc['lens'] = struct.unpack('h', f.read(2))[0]
        mrc['nd1'] = struct.unpack('h', f.read(2))[0]
        mrc['nd2'] = struct.unpack('h', f.read(2))[0]
        mrc['vd1'] = struct.unpack('h', f.read(2))[0]
        mrc['vd2'] = struct.unpack('h', f.read(2))[0]
        mrc['tiltangles'] = struct.unpack(('f' * 6), f.read((4 * 6)))
        mrc['xorg'] = struct.unpack('f', f.read(4))[0]
        mrc['yorg'] = struct.unpack('f', f.read(4))[0]
        mrc['zorg'] = struct.unpack('f', f.read(4))[0]
        mrc['cmap'] = struct.unpack(('c' * 4), f.read(4))
        mrc['stamp'] = struct.unpack(('c' * 4), f.read(4))
        mrc['rms'] = struct.unpack('f', f.read(4))[0]
        mrc['nlabl'] = struct.unpack('i', f.read(4))[0]
        mrc['labl'] = struct.unpack(('c' * 800), f.read(800))
        size = [mrc['nx'], mrc['ny'], mrc['nz']]
        n_voxel = N.prod(size)
        extended = {}
        extended['magnification'] = [0]
        extended['exp_time'] = [0]
        extended['pixelsize'] = [0]
        extended['defocus'] = [0]
        extended['a_tilt'] = ([0] * mrc['nz'])
        extended['tiltaxis'] = [0]
        if (mrc['next'] != 0):
            nbh = (mrc['next'] / 128)
            if (nbh == 1024):
                for lauf in range(nbh):
                    extended['a_tilt'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['b_tilt'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['x_stage'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['y_stage'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['z_stage'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['x_shift'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['y_shift'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['defocus'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['exp_time'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['mean_int'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['tiltaxis'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['tiltaxis'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['pixelsize'][lauf] = struct.unpack('f', f.read(4))[0]
                    extended['magnification'][lauf] = struct.unpack('f', f.read(4))[0]
                    f.seek(offset=(128 - 52), whence=1)
                else:
                    f.seek(offset=MRC.__next__, whence=1)
        if read_data:
            slice_voxel_num = (mrc['nx'] * mrc['ny'])
            v = None
            for i in range(mrc['nz']):
                if show_progress:
                    print('\r', i, '   ', end=' ')
                    sys.stdout.flush()
                if (mrc['mode'] == 0):
                    if (v is None):
                        v = N.zeros(size, dtype=N.int8)
                    data_read = N.fromfile(f, dtype=N.int8, count=slice_voxel_num)
                elif (mrc['mode'] == 1):
                    if (v is None):
                        v = N.zeros(size, dtype=N.int16)
                    data_read = N.fromfile(f, dtype=N.int16, count=slice_voxel_num)
                elif (mrc['mode'] == 2):
                    if (v is None):
                        v = N.zeros(size, dtype=N.float32)
                    data_read = N.fromfile(f, dtype=N.float32, count=slice_voxel_num)
                else:
                    raise Exception('Sorry, i cannot read this as an MRC-File !!!')
                    data_read = None
                if (data_read.size != slice_voxel_num):
                    import pdb
                    pdb.set_trace()
                v[:, :, i] = N.reshape(data_read, (mrc['nx'], mrc['ny']), order='F')
        else:
            v = None
        h = {}
        h['Voltage'] = None
        h['Cs'] = None
        h['Aperture'] = None
        h['Magnification'] = extended['magnification'][0]
        h['Postmagnification'] = None
        h['Exposuretime'] = extended['exp_time'][0]
        h['Objectpixelsize'] = (extended['pixelsize'][0] * 1000000000.0)
        h['Microscope'] = None
        h['Pixelsize'] = None
        h['CCDArea'] = None
        h['Defocus'] = extended['defocus'][0]
        h['Astigmatism'] = None
        h['AstigmatismAngle'] = None
        h['FocusIncrement'] = None
        h['CountsPerElectron'] = None
        h['Intensity'] = None
        h['EnergySlitwidth'] = None
        h['EnergyOffset'] = None
        h['Tiltangle'] = extended['a_tilt'][:mrc['nz']]
        h['Tiltaxis'] = extended['tiltaxis'][0]
        h['Username'] = None
        h['Date'] = None
        h['Size'] = [mrc['nx'], mrc['ny'], mrc['nz']]
        h['Comment'] = None
        h['Parameter'] = None
        h['Fillup'] = None
        h['Filename'] = path
        h['Marker_X'] = None
        h['Marker_Y'] = None
        h['MRC'] = mrc
    return {'header': h, 'value': v, }

def read_mrc_vol(path, show_progress=False):
    return read_mrc(path=path, show_progress=show_progress)['value']

def put_mrc(mrc, path, overwrite=True):
    path = os.path.realpath(path)
    if (mrc.dtype != N.float):
        mrc = N.array(mrc, order='F', dtype=N.float)
    if (not mrc.flags['F_CONTIGUOUS']):
        mrc = N.array(mrc, order='F', dtype=N.float)
    path = str(path)
    if ((overwrite == False) and os.path.isfile(path)):
        return
    import aitom.tomominer.core.core as core
    core.write_mrc(mrc, path)

