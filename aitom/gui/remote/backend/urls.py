from django.urls import path
from . import model3d, slice

'''
all the backend url have prefix "api/"
'''


urlpatterns = [ 
    path('model', model3d.process),
    path('slice', slice.process),
] 