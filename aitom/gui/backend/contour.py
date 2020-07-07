from mayavi import mlab
import numpy as np
import mrcfile
from mayavi.mlab import contour3d, volume_slice
from tvtk.api import write_data


def process(path, savepath):
    mrc = mrcfile.open(path)
    print(type(mrc.data))
    obj = contour3d(mrc.data, contours=4)
    vtkout = obj.contour.contour_filter.output
    write_data(vtkout, savepath)
    # obj.module_manager.source.save_output(savepath)
    return obj


if __name__ == '__main__':
    obj = process('test.mrc', 'test.vtk')
    # vtkout = obj.contour.contour_filter.output
    # write_data(vtkout, 'test.vtk')
    # print(type(vtkout), vtkout)
    # vtk = mlab.pipeline.get_vtk_src(obj)
    # print(vtk)
    # mlab.show()
