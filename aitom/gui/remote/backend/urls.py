"""
all the backend url have prefix "api/"
"""

from django.urls import path
from .autoencoder import particle_picking, particle_picking_visualization, \
    autoencoder_single, particle_picking_resume, autoencoder_result
from .model3d import process as model_process
from .model3d import process_json as model_process_json
from .slice import process as slice_process


urlpatterns = [
    path('model', model_process, name='api-model'),
    path('model-json', model_process_json),
    path('slice', slice_process),
    path('particle-picking', particle_picking),
    path('particle-picking-visualization', particle_picking_visualization),
    path('particle-picking-resume', particle_picking_resume),
    path('autoencoder-single', autoencoder_single),
    path('autoencoder-result', autoencoder_result)
]
