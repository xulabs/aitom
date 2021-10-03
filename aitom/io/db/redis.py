

'''
a proxy to redis in-memory data structure store

'''

import cPickle as pickle
import redis

class Redis:
    def __init__(self, host='localhost', port=6379, db=0, expire_time=None):
        self.redis = redis.StrictRedis(host=host, port=port, db=db)
        self.expire_time = expire_time

    def set(self, k, v):
        r = self.redis.set(k, pickle.dumps(v, protocol=-1))
        if self.expire_time is not None:        self.redis.expire(k, self.expire_time)
        return r


    def get(self, k):
        r = pickle.loads(self.redis.get(k))
        if self.expire_time is not None:        self.redis.expire(k, self.expire_time)
        return r

    def get_del(self, k):
        r = pickle.loads(self.redis.get(k))
        self.redis.delete(k)
        return r


    def delete(self, k):
        return self.redis.delete(k)

    def expire(self, k):
        return self.redis.expire(k)

    def save(self):
        return self.redis.save()

    def bgsave(self):
        return self.redis.bgsave()


