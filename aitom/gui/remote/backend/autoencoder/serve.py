from ..base import request_check
from .proto import *
from .particle_picking import particle_picking_main, \
    particle_picking_visualization_main, particle_picking_resume_main
from .autoencoder import autoencoder_single_main, autoencoder_result_main
from django.http import HttpRequest, Http404, HttpResponse


@request_check(methods={'POST'})
def particle_picking(request):
    response = particle_picking_main(PPRequest().deserialize(request))  # type: PPResponse
    return HttpResponse(response.serialize(), status=200)


@request_check(methods={'POST'})
def particle_picking_resume(request):
    response = particle_picking_resume_main(PPResumeRequest().deserialize(request))
    return HttpResponse(response.serialize(), status=200)


@request_check(methods={'POST'})
def particle_picking_visualization(request):
    response = particle_picking_visualization_main(PPVisRequest().deserialize(request))  # type: PPVisResponse
    return HttpResponse(response.serialize(), status=200)


@request_check(methods={'POST'})
def autoencoder_single(request):
    response = autoencoder_single_main(AESingleRequest().deserialize(request))  # type: AESingleResponse
    return HttpResponse(response.serialize(), status=200)


@request_check(methods={'POST'})
def autoencoder_result(request):
    response = autoencoder_result_main(AEResultRequest().deserialize(request))  # tyoe: AEResultResponse
    return HttpResponse(response.serialize(), status=200)
