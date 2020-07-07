from mayavi import mlab
import numpy as np
import mrcfile
from mayavi.mlab import contour3d, volume_slice
from tvtk.api import write_data
from django.conf import settings


def process(path, savepath):

    path = settings.MEDIA_ROOT + '/' + path #comment this and line 15 and correct line 17 if executing only locally. This is configuration for the server.
    mrc = mrcfile.open(path)
   # print(type(mrc.data))
    obj = contour3d(mrc.data, contours=4,figure=None)
    mlab.close(all=True)
    vtkout = obj.contour.contour_filter.output
    write_data(vtkout, settings.MEDIA_ROOT+ '/' + savepath)
    # obj.module_manager.source.save_output(savepath)
    return obj
    


if __name__ == '__main__':
    obj = process('test.mrc', 'test.vtk')
    
