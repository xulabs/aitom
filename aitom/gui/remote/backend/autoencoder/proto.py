import sys
from ..base import BaseResponseProto, BaseRequestProto


class PPRequest(BaseRequestProto):
    path_key = 'path'
    # output_dir_key = 'output'
    sigma1_key = 'sigma1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = ''
        self.output_dir = r'./tmp/picking'
        self.sigma1 = -1


class PPResponse(BaseResponseProto):
    pick_total_key = 'pick_total'
    uid_key = 'uid'

    # dump_path_key = 'dump_path'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for particle picking
        self.pick_total = -1
        self.uid = -1
        # self.dump_path = ''


class PPVisRequest(BaseRequestProto):
    subvol_num_key = 'subvol_num'
    path_key = 'path'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subvol_num = -1
        self.path = ''


class PPVisResponse(BaseResponseProto):
    subvol_url_key = 'subvol_url'

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.subvol_url = ''


class PPResumeRequest(BaseRequestProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PPResumeResponse(BaseResponseProto):
    resumable_list_key = 'resumable_list'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resumable_list = []


class AESingleRequest(BaseRequestProto):
    path_key = 'path'
    remove_particles_key = 'remove_particles'

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.path = ''
        self.remove_particles = ''


class AESingleResponse(BaseResponseProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AETrainingRequest(BaseRequestProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    pass


class AETrainingResponse(BaseResponseProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    pass


class AEResultRequest(BaseRequestProto):
    path_key = 'path'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = ''


class AEResultResponse(BaseResponseProto):
    img_key = 'img'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img = ''
