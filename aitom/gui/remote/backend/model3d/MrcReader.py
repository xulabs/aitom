import mrcfile
import os
from .MrcWriter import MrcWriter


class MrcReader:
    def __init__(self, path, scale_rate=0):
        self.base_path = path
        self.scale_rate = scale_rate
        self.path = self.get_path(self.base_path, self.scale_rate)
        self.next = None
        self.mrc = mrcfile.mmap(self.path, 'r')
        self._check_scale()

    def _check_scale(self):
        scale_path = self.get_path(self.base_path, self.scale_rate + 1)

        shape = self.mrc.data.shape
        if any([x <= 50 for x in shape]):
            return
        if not os.path.exists(scale_path):
            MrcWriter.scale(self.path, scale_path, 1)
            self.next = MrcReader(self.base_path, self.scale_rate + 1)

    @staticmethod
    def get_path(base_path: str, scale_rate):
        if scale_rate == 0:
            return base_path
        else:
            index = base_path.rfind('.')
            scale_path = base_path[:index] + f'_{scale_rate}' + base_path[index:]
            return scale_path

    def __del__(self):
        self.mrc.close()

    def __getitem__(self, key):
        return self.mrc.data[key]


if __name__ == '__main__':
    m = MrcReader('test.mrc')
