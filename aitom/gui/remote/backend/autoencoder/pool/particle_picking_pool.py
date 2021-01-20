import os
import pickle
from ..particle_picking_and_autoencoder_util import ParticlePicking
from ..proto import *
from typing import Optional

PROJECT_APP_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_APP_PATH = os.path.abspath(os.path.join(PROJECT_APP_PATH, os.pardir, os.pardir, os.pardir))
MRC_LIBRARY_PATH = os.path.join(PROJECT_APP_PATH, 'library', 'mrc')
PP_DUMP_PATH = os.path.join(PROJECT_APP_PATH, 'library', 'pp')
if not os.path.exists(PP_DUMP_PATH):
    os.mkdir(PP_DUMP_PATH)


class ParticlePickingPoolItem:
    dump_name = 'item.pickle'

    def __init__(self, name, uid):
        self.mrc_name = name
        self.uid = int(uid)
        self.mrc_path = os.path.join(MRC_LIBRARY_PATH, name)
        self.dump_folder = os.path.join(PP_DUMP_PATH, str(uid))
        self.dump_path = os.path.join(
            self.dump_folder, self.dump_name)
        self.proto_dict = {}
        self.pick = None

        if not os.path.exists(self.dump_folder):
            os.mkdir(self.dump_folder)
            self.pick = ParticlePicking(self.mrc_path, self.dump_folder)
        else:
            if os.path.exists(self.dump_path):
                self.load_item()
            else:
                self.pick = ParticlePicking(self.mrc_path, self.dump_folder)

    def add_proto(self, proto):
        proto_type = type(proto)
        if proto_type not in self.proto_dict:
            self.proto_dict[proto_type] = []
        self.proto_dict[proto_type].append(proto)
        self.dump_item()

    def load_item(self):
        with open(self.dump_path, 'rb') as f:
            obj = pickle.load(f)
        for k, v in obj.__dict__.items():
            setattr(self, k, v)

    def dump_item(self):
        with open(self.dump_path, 'wb') as f:
            pickle.dump(self, f)


class ParticlePickingPool:
    def __init__(self, max_num=20, clean_num=5):
        assert clean_num < max_num, 'clean num should be lower than max num'
        self.max_num = max_num
        self.clean_num = clean_num
        self.pool = {}
        self.__load()

    def get(self, name: str, new_one=False) -> Optional[ParticlePickingPoolItem]:
        print('fetching pool', name)
        if name.isdigit():
            # treat name as uid
            return self.pool[int(name)]
        else:
            if new_one:
                return self.new(name)
        return None

    def new(self, name: str):
        if len(self.pool) < self.max_num:
            for i in range(self.max_num):
                if i not in self.pool:
                    self.pool[i] = ParticlePickingPoolItem(name, i)
                    return self.pool[i]
        else:
            self.__clean()
            return self.new(name)
        return None

    def make_list(self):
        ret = []
        for k, v in self.pool.items():
            sigma = v.proto_dict.get(PPRequest, None)
            if sigma is not None:
                sigma = sigma[-1].sigma1
            ret.append({
                'uid': k,
                'mrc_name': v.mrc_name,
                'sigma': sigma
            })
        return ret
    
    def __load(self):
        for i in range(self.max_num):
            folder = os.path.join(PP_DUMP_PATH, str(i))
            if not os.path.exists(folder):
                continue
            dump_item = ParticlePickingPoolItem('None', i)
            dump_item.load_item()
            self.pool[i] = dump_item

    def __clean(self):
        remove = []
        for i in range(self.max_num):
            folder = os.path.join(PP_DUMP_PATH, str(i))
            item_path = os.path.join(folder, ParticlePickingPoolItem.dump_name)
            modify_time = os.stat(item_path).st_mtime
            remove.append((modify_time, folder))
        remove.sort(key=lambda x: x[0])
        for i in range(self.clean_num):
            os.removedirs(remove[i][1])
            self.pool.pop(i)


particlePickingPool = ParticlePickingPool()
