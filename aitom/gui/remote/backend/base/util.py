from django.http import HttpRequest, HttpResponse
from django.http import HttpResponseNotAllowed


def request_check(methods={'POST'}):
    """
    general check for requests
    """

    def real_decorator(fn):
        def wrapper(request: HttpRequest, *args, **kwargs):
            if request.method not in methods:
                return HttpResponseNotAllowed(methods)
            return fn(request, *args, **kwargs)

        return wrapper

    return real_decorator
