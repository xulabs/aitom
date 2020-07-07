from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from mayavi import mlab
import numpy as np
import mrcfile
from mayavi.mlab import contour3d, volume_slice
from tvtk.api import write_data
import os
from django.views.decorators.clickjacking import xframe_options_exempt
from django.conf import settings
from backend.contour import process  
PROJECT_APP_PATH = os.path.dirname(os.path.abspath(__file__))
def index(request):

     if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save('test.mrc', myfile.file)
        obj = process(filename, filename.replace(".mrc",".vtk"))

       # filename = fs.save('test.vtk', obj)
     #   uploaded_file_url = fs.url(filename)
        uploaded_file_url = '/mrc/' + filename.replace(".mrc",".vtk")
        return render(request, PROJECT_APP_PATH + '/templates/gui/index.html', {
            'uploaded_file_url': uploaded_file_url
        })
     return render(request,PROJECT_APP_PATH + '/templates/gui/index.html')
    
@xframe_options_exempt
 
def display(request):
     
     return render(request,PROJECT_APP_PATH + '/templates/gui/display.html')     
     



def process(path, savepath):
    path = settings.MEDIA_ROOT + '/' + path
    mrc = mrcfile.open(path)
   # print(type(mrc.data))
    obj = contour3d(mrc.data, contours=4,figure=None)
    mlab.close(all=True)
    vtkout = obj.contour.contour_filter.output
    write_data(vtkout, settings.MEDIA_ROOT+ '/' + savepath)
    # obj.module_manager.source.save_output(savepath)
    return obj
    
   
