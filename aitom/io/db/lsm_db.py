"""
wrapper for LSM / SQLite4
with pickle for storing values of arbitrary value types

pip install lsm-db
"""

import pickle as pickle


class LSM:
    def __init__(self, filename, readonly=False):
        # from the lsm-db package
        import lsm
        self.db = lsm.LSM(filename=filename, readonly=readonly)

    def __getitem__(self, key):
        value = self.db[self.__keytransform__(key)]
        value = pickle.loads(value)
        return value

    def __setitem__(self, key, value):
        value = pickle.dumps(value, protocol=-1)
        self.db[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.db[self.__keytransform__(key)]

    def __iter__(self):
        return iter(list(self.db.keys()))

    def __len__(self):
        # get the length by retriving all keys, this is a slow operation, need to find a faster way
        return len([_ for _ in list(self.db.keys())])

    def __contains__(self, key):
        return key in self.db

    def __keytransform__(self, key):
        return key

    def keys(self):
        return iter(list(self.db.keys()))
