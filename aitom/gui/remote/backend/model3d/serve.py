from django.http import HttpRequest, Http404
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import json
from ..util import request_check
from .contour import contour
from urllib.parse import urljoin
import os.path

PROJECT_APP_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_APP_PATH = os.path.abspath(os.path.join(PROJECT_APP_PATH, os.pardir)) #1 level parent
PROJECT_APP_PATH = os.path.abspath(os.path.join(PROJECT_APP_PATH, os.pardir)) #2 level parent (root directory)
def process(request: HttpRequest):
    #check = request_check(request)
    #if check: return check


    if request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save('test.mrc', myfile.file)
        obj = contour(filename, filename.replace(".mrc",".vtk"))
        print('here')
        #filename = fs.save('test.vtk', obj)
     #   uploaded_file_url = fs.url(filename)
        uploaded_file_url = urljoin('/uploads/', filename.replace(".mrc",".vtk"))
        print(uploaded_file_url)
        return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/index.html', {
            'uploaded_file_url': uploaded_file_url
        })
