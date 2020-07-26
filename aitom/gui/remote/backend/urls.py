from django.urls import path
from . import model3d, slice

'''
all the backend url have prefix "api/"
'''


<<<<<<< HEAD
urlpatterns = [
    path('model', model3d.process,name='api-model'),
    path('model-json', model3d.process_json),
    path('slice', slice.process),
]
=======
urlpatterns = [ 
    path('model', model3d.process),
    path('slice', slice.process),
] 
>>>>>>> b75fc973a6f53ef3bcccca31402135ce7c7b5b6d
