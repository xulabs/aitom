from django.http import HttpRequest, HttpResponse
from ..util import request_check
from .slice import slice3d
from io import BytesIO
import ast
import os
import mrcfile

PROJECT_APP_PATH = os.path.dirname(os.path.abspath(__file__))
# 1 level parent
PROJECT_APP_PATH = os.path.abspath(os.path.join(PROJECT_APP_PATH, os.pardir))
# 2 level parent (root directory)
PROJECT_APP_PATH = os.path.abspath(os.path.join(PROJECT_APP_PATH, os.pardir))


def process(request: HttpRequest):
    check = request_check(request)
    if check:
        return check
    # post request values to useable format
    req = list(dict(request.POST).keys())[0]

    req.replace("[[", "[")
    req.replace("]]", "]")
    req = ast.literal_eval(req)[0]

    getKey = lambda key: next(item for item in req if item["name"] == key)['value']

    # extract data from json
    try:
        file_name = getKey('filename')
        method = int(getKey('method'))
        center = [int(getKey('cX')), int(getKey('cY')), int(getKey('cZ'))]
        def_plane = getKey('plane')
        rotation = {'x': float(getKey('rX')), 'y': float(getKey('rY')), 'z': float(getKey('rZ'))}
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
    # print(abs_file_path)
    if not os.path.exists(abs_file_path):  # check file exists
        return HttpResponse('file not exists on server', status=400)
    # if any([lu.x >= rd.x, lu.y >= rd.y, lu.z >= rd.z]): # check point
    #    return HttpResponse('left-up point should be smaller than right-down point', status=400)

    try:
        mrc = mrcfile.open(abs_file_path)
        model = mrc.data
        # ret_slice = BytesIO()
        ret_slice = slice3d(model, center, rotation['x'], rotation['y'], rotation['z'], def_plane)
        # print(ret_slice)
        return HttpResponse(ret_slice, status=200)

    except Exception as e:
        return HttpResponse('error occured when processing files:{}'.format(str(e)), status=400)
