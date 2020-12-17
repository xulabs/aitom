import mrcfile
import numpy as np
import os
import sys

sys.path.append("..")


def map2mrc(map, file):
    with mrcfile.new(file, overwrite=True) as mrc:
        mrc.set_data(map.astype(np.float32))


def map2npy(map, file):
    np.save(file, map)


def map2png(map, file):
    import aitom.image.io as IIO
    import aitom.image.vol.util as TIVU
    IIO.save_png(TIVU.cub_img(map)['im'], file)


def readMrcMap(file):
    if file.endswith(".mrc"):
        with mrcfile.open(file) as mrc:
            return mrc.data


def readMrcMapDir(dir):
    # read density map in a dir
    list = os.listdir(dir)
    v = {}
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isfile(path):
            (filename, extension) = os.path.splitext(list[i])
            if path.endswith('.mrc'):
                with mrcfile.open(path) as mrc:
                    v[filename] = mrc.data
    return v
