from django.http import HttpRequest, HttpResponse
from ..util import request_check

def process(request: HttpRequest):
    check = request_check(request)
    if check: return check

    return HttpResponse('slice test')