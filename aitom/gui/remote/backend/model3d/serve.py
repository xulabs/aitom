from django.http import HttpRequest, Http404, HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import json
from ..util import request_check
from .contour import contour
from .MrcLoader import MrcLoader
from urllib.parse import urljoin
import os.path
import ast
from django_server.models import Document
from django_server.forms import DocumentForm
from django2_resumable.files import ResumableFile, get_storage, get_chunks_upload_to

from collections import namedtuple

Point = namedtuple('Point', ['x', 'y', 'z'])

PROJECT_APP_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_APP_PATH = os.path.abspath(os.path.join(PROJECT_APP_PATH, os.pardir))  # 1 level parent
PROJECT_APP_PATH = os.path.abspath(os.path.join(PROJECT_APP_PATH, os.pardir))  # 2 level parent (root directory)


def process(request):
    upload_to = get_chunks_upload_to(request)
    storage = get_storage(upload_to)
    if request.method == 'POST':
        chunk = request.FILES.get('file')
        r = ResumableFile(storage, request.POST)
        if not r.chunk_exists:
            r.process_chunk(chunk)
        if r.is_complete:
            filename = str(storage.save(r.filename, r.file))
            r.delete_chunks()
            # print(filename)
            base = PROJECT_APP_PATH + '/uploads'
            base_temp = PROJECT_APP_PATH + '/temp'
            filepath_mrc = base + '/mrc/' + filename
            # No need to convert a newly uploaded MRC file to VTK at this stage. We will do it later when /api/json
            # is called
            # filepath_vtk = base + '/vtk/' + filename.replace(".mrc",".vtk")
            # obj = contour(filepath_mrc, filepath_vtk)
            # delete the uploaded file
            # p = Popen("rm %s" % filepath_mrc, shell=True)
            # uploaded_file_url = urljoin(filepath_vtk)
            # print(uploaded_file_url)

            return HttpResponse(filename, status=201)
        return HttpResponse('chunk uploaded')


def process_json(request: HttpRequest):
    check = request_check(request)
    # post request values to useable format
    req = list(dict((request.POST)).keys())[0]

    req.replace("[[", "[")
    req.replace("]]", "]")
    req = ast.literal_eval(req)[0]

    getKey = lambda key: next(item for item in req if item["name"] == key)['value']

    if check:
        return check

    # extract data from json
    try:
        file_name = getKey('filename')
        method = int(getKey('method'))
        print(method)
        lu = Point(*map(int, (getKey('luX'), getKey('luY'), getKey('luZ'))))
        rd = Point(*map(int, (getKey('rdX'), getKey('rdY'), getKey('rdZ'))))
    except Exception as e:
        return HttpResponse('key error: {}'.format(str(e)), status=400)

    # check data
    base_mrc_folder = os.path.join(PROJECT_APP_PATH, 'library', 'mrc')
    base_temp_vtk_folder = os.path.join(PROJECT_APP_PATH, 'temp', 'vtk')
    base_temp_mrc_folder = os.path.join(PROJECT_APP_PATH, 'temp', 'mrc')
    abs_file_path = ''
    if method == 1:
        abs_file_path = os.path.join(PROJECT_APP_PATH, 'library', 'mrc', file_name)
    elif method == 2:
        abs_file_path = os.path.join(PROJECT_APP_PATH, 'uploads', 'mrc', file_name)
    print(abs_file_path)
    if not os.path.exists(abs_file_path):  # check file exists
        return HttpResponse('file not exists on server', status=400)
    if any([lu.x >= rd.x, lu.y >= rd.y, lu.z >= rd.z]):  # check point
        return HttpResponse('left-up point should be smaller than right-down point', status=400)

    try:
        scaled_mrc_name = MrcLoader(abs_file_path).read(lu, rd, scale=0, base_path=base_temp_mrc_folder)
        obj_path = os.path.join(base_temp_vtk_folder, scaled_mrc_name.replace('.mrc', '.vtk'))
        contour(os.path.join(base_temp_mrc_folder, scaled_mrc_name), obj_path)
        uploaded_file_url = urljoin('/temp/vtk/', scaled_mrc_name.replace('.mrc', '.vtk'))
    except Exception as e:
        return HttpResponse('error occured when processing files:{}'.format(str(e)), status=400)

    return HttpResponse(uploaded_file_url, status=201)
