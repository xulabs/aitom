from django.http import HttpRequest, HttpResponse
from django.http import HttpResponseNotAllowed


def request_check(request: HttpRequest, methods={'POST'}):
    """
    general check for requests
    """
    if request.method not in methods:
        return HttpResponseNotAllowed(methods)
