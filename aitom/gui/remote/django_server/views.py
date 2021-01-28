from django.shortcuts import render
from django.conf import settings

import os.path
from django.views.decorators.clickjacking import xframe_options_exempt
from django.conf import settings
from django_server.models import Document
from django_server.forms import DocumentForm, zoomForm, sliceForm
from django.http import StreamingHttpResponse, Http404
from wsgiref.util import FileWrapper
from django import forms
import glob
import json
import ast

# /remote/django_server
PROJECT_APP_PATH = os.path.dirname(os.path.abspath(__file__))
# /remote
PROJECT_APP_PATH = os.path.abspath(os.path.join(PROJECT_APP_PATH, os.pardir))


def index(request):
    # form = DocumentForm()
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/index.html')


@xframe_options_exempt
# just serve display.html, without any extra fields. xframe options enabled for display and inst1,2,3 to allow
# embedding into an iframe
def display(request):
    # the display loader with three.js on the frontend
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/display.html')


@xframe_options_exempt
def disp_img(request):
    # the display loader with three.js on the frontend
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/disp-img.html')


@xframe_options_exempt
def inst1(request):
    # introduction and first instruction page
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/inst1.html')


@xframe_options_exempt
def inst2(request):
    # second instruction page
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/inst2.html')


@xframe_options_exempt
def inst3(request):
    # third instruction page
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/inst3.html')


@xframe_options_exempt
def inst4(request):
    # third instruction page
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/inst4.html')


def particle_picking_index(request):
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/particle-picking.html', {
        'form': MyForm(),
    })


@xframe_options_exempt
def pp_inst(request, inst_index):
    try:
        inst_index = int(inst_index)
    except ValueError:
        return Http404()
    html_path = PROJECT_APP_PATH + '/frontend/templates/frontend/particle-picking-inst/inst{}.html'.format(inst_index)
    if not os.path.exists(html_path):
        return Http404
    return render(request, html_path)


# Function for serving VTK files chunk by chunk, this method is more efficient than loading entire VTK file into
# memory at once
# Implemented with help from: https://stackoverflow.com/questions/43591440/django-1-11-download-file-chunk-by-chunk
def download(request):
    file_path = PROJECT_APP_PATH + request.GET["path"]
    # print(file_path)
    # DEFINE_A_CHUNK_SIZE_AS_INTEGER
    chunk_size = 8192
    filename = os.path.basename(file_path)
    # print(filename)
    response = StreamingHttpResponse(
        FileWrapper(open(file_path, 'rb'), chunk_size),
        content_type="application/octet-stream"
    )
    response['Content-Length'] = os.path.getsize(file_path)
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response


# Serve model form with resumeable.js enabled upload field
def getUploadForm(request):
    form = DocumentForm()
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/upload.html', {'form': form})


class MyForm(forms.Form):
    names = []

    # Get a list of MRC models stored in library folder using glob
    documents = glob.glob(PROJECT_APP_PATH + "/library/mrc/*.mrc")

    for doc in documents:
        name = doc.split('/')[-1].split('\\')[-1]  # in different OS
        names.append([name, name])
    # build a <select> element with file names as options derived from above for loop
    select = forms.ChoiceField(widget=forms.Select, choices=names)


def getLibrary(request):
    # serve the select field built above
    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/list.html', {
        'form': MyForm()
    })


def getInputForm(request):
    # serve the form for inputting values to zoom into
    req = list(dict(request.POST).keys())[0]  # convert post request values to useable format
    # using this non standard approach cause I can't figure out why Query Dict is not loading properly with AJAX json
    # requests

    req.replace("[[", "[")
    req.replace("]]", "]")
    req = ast.literal_eval(req)  # covert string to dictionary using ast
    op = int(req['op'])

    if op == 1:  # zoom
        form = zoomForm()
        url = '/api/model-json'
    elif op == 2:  # slice
        form = sliceForm()
        url = '/api/slice'

    return render(request, PROJECT_APP_PATH + '/frontend/templates/frontend/input.html', {
        'form': form, 'filename': req["filename"],
        'method': req["method"],
        'op': req["op"],
        'request_url': url
    })
