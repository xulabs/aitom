

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import os
import stat
import shutil
import time
import uuid
import pickle
import numpy as N
import aitom.tomominer.io.file as IF


class Cache:

    def __init__(self, cache_dir=None, tmp_dir=None, logger=None):
        self.logger = logger
        self.cache_dir = cache_dir
        self.tmp_dir = tmp_dir

    def get_temp_file_path(self, prefix=None, fn_id=None, suffix=None, ext=None):
        if (fn_id is not None):
            fn = fn_id
        else:
            fn = str(uuid.uuid4())
        if (prefix is not None):
            fn = (prefix + fn)
        if (suffix is not None):
            fn = (fn + suffix)
        if (ext is not None):
            fn += ext
        assert (self.tmp_dir is not None)
        return os.path.join(self.tmp_dir, fn)

    def save_tmp_data(self, d, fn_id=None):
        key = self.get_temp_file_path(prefix='tmp-', fn_id=fn_id, ext='.pickle')
        assert (not os.path.isfile(key))
        with open(key, 'wb') as f:
            pickle.dump(d, f, protocol=(-1))
        os.chmod(key, 438)
        return key

    def get_mrc(self, path):
        path = str(path)
        return self.get_mrc_cache_fs(path)

    def get_mrc_cache_fs(self, path):
        return self.load_file_cache_fs(IF.read_mrc_vol, path)

    def load_file_cache_fs(self, load_func, path):
        path = os.path.abspath(path)
        if (self.cache_dir is None):
            v = load_func(path)
            return v
        if (not os.path.isdir(self.cache_dir)):
            try:
                os.makedirs(self.cache_dir)
            except:
                pass
            if (not os.path.isdir(self.cache_dir)):
                raise OSError(('cache_dir   ' + self.cache_dir))
        cache_path = (self.cache_dir + path)
        while self.load_file_cache_fs__is_miss(path=path, cache_path=cache_path):
            cache_path__dir = os.path.dirname(cache_path)
            if (not os.path.isdir(cache_path__dir)):
                try:
                    os.makedirs(cache_path__dir)
                except:
                    pass
            shutil.copyfile(path, cache_path)
            time.sleep(1)
        v = load_func(cache_path)
        return v

    def load_file_cache_fs__is_miss(self, path, cache_path):
        miss = False
        if (not os.path.isfile(cache_path)):
            miss = True
        else:
            path__st = os.stat(path)
            cache_path__st = os.stat(cache_path)
            if (path__st.st_mtime > cache_path__st.st_mtime):
                miss = True
            if (path__st.st_size != cache_path__st.st_size):
                miss = True
        return miss