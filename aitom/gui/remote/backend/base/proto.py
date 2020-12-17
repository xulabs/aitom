from abc import abstractmethod
import json


class BaseRequestProto:
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def deserialize(self, request, *args, **kwargs):
        json_dict = request.POST
        for k, v in type(self).__dict__.items():
            if not k.startswith('_') and k[-4:] == '_key':
                as_type = type(getattr(self, k[:-4]))
                setattr(self, k[:-4], as_type(json_dict[v]))
        return self


class BaseResponseProto:
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def serialize(self, *args, **kwargs):
        d = {}
        for k, v in type(self).__dict__.items():
            if not k.startswith('_') and k[-4:] == '_key':
                d[v] = getattr(self, k[:-4])
        return json.dumps(d)
