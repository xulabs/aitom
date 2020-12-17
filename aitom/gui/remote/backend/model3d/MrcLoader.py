import mrcfile
from .MrcReader import MrcReader
import zipfile
import os
from random import randint


class MrcLoader:
    def __init__(self, path):
        self.path = path
        origin_reader = MrcReader(self.path)
        self.reader_list = []

        r = origin_reader
        while r is not None:
            self.reader_list.append(r)
            r = r.next

    def read(self, leftDown, rightUp, scale, base_path='', require_zip=False):
        x0, y0, z0 = leftDown
        x1, y1, z1 = rightUp
        temp_data = self.reader_list[scale][x0: x1, y0: y1, z0: z1]
        temp_path = 'temp{}.mrc'.format(randint(1, 10000))
        with mrcfile.new(os.path.join(base_path, temp_path), overwrite=True) as temp_mrc:
            temp_mrc.set_data(temp_data)
        if not require_zip:
            return temp_path
        else:
            return self._zipmrc(temp_path)

    def _zipmrc(self, mrc):
        z = zipfile.ZipFile(mrc + '.zip', 'w', zipfile.ZIP_DEFLATED)
        z.write(mrc)
        z.close()
        os.remove(mrc)
        return mrc + '.zip'
