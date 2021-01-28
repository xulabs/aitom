import mrcfile
import numpy as np
from .config import SCALE_BASE
from tqdm import tqdm


class MrcWriter:
    @staticmethod
    def scale(origin_path, new_path, scale_rate=1):
        origin_mrc = mrcfile.mmap(origin_path, 'r')
        origin_shape = np.array(origin_mrc.data.shape)
        ox, oy, oz = origin_shape
        scale = SCALE_BASE ** scale_rate
        new_shape = (origin_shape + scale - 1) // scale
        nx, ny, nz = new_shape
        new_mrc = mrcfile.new_mmap(new_path, (nz, nx, ny),
                                   mrc_mode=mrcfile.utils.mode_from_dtype(
                                       mrcfile.utils.data_dtype_from_header(origin_mrc.header)
                                   ),
                                   overwrite=True)
        print('scaling data')
        for z in tqdm(range(nz)):
            sz = z * scale
            for x in range(nx):
                sx = x * scale
                for y in range(ny):
                    sy = y * scale
                    new_mrc.data[z, x, y] = np.mean(origin_mrc.data[sz: min(sz + scale, oz),
                                                    sx: min(sx + scale, ox),
                                                    sy: min(sy + scale, oy)
                                                    ])
        print('updating header')
        new_mrc.update_header_stats()
        print('header updated')
        origin_mrc.close()
        new_mrc.close()

    @staticmethod
    def write(path, data):
        raise NotImplementedError


if __name__ == '__main__':
    MrcWriter.scale('t2.mrc', 't3.mrc')
