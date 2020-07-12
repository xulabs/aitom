from django.shortcuts import render
from django.conf import settings

import os.path
from django.views.decorators.clickjacking import xframe_options_exempt
from django.conf import settings

PROJECT_APP_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_APP_PATH = os.path.abspath(os.path.join(PROJECT_APP_PATH, os.pardir))

def index(request):
    return render(request,PROJECT_APP_PATH + '/frontend/templates/frontend/index.html')
    
@xframe_options_exempt
 
def display(request):
     
    return render(request,PROJECT_APP_PATH + '/frontend/templates/frontend/display.html')     
     




   
