"""
gui URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, re_path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls import url, include

# expose the required URLs
# a script with 'api' prefix goes to backend for handling
# all other url are handled by front end modules
urlpatterns = [
                  path('', views.index, name='index'),
                  path('inst1/', views.inst1, name='inst1'),
                  path('inst2/', views.inst2, name='inst2'),
                  path('inst3/', views.inst3, name='inst3'),
                  path('inst4/', views.inst4, name='inst4'),
                  path('display/', views.display, name='display'),
                  path('disp-img/', views.disp_img, name='disp_img'),
                  path('download/', views.download, name='download'),
                  path('admin/', admin.site.urls),
                  path('api/', include('backend.urls')),
                  path('getUploadForm/', views.getUploadForm, name='getUploadForm'),
                  path('getLibrary/', views.getLibrary, name='getLibrary'),
                  path('getInputForm/', views.getInputForm, name='getInputForm'),
                  path('particle-picking/', views.particle_picking_index, name='particle-picking'),
                  re_path(r'^particle-picking-inst/inst(\d+)', views.pp_inst, name='particle-picking-inst')
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
