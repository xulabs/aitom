

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import uuid
import time


class Task(object):

    def __init__(self, priority=1000, proj_id=None, module=None, method=None, args=[], kwargs={}):
        self.priority = priority
        self.proj_id = proj_id
        assert (proj_id is not None)
        self.task_id = str(uuid.uuid4())
        self.module = module
        assert (module is not None)
        self.method = method
        assert (method is not None)
        self.args = args
        self.kwargs = kwargs
        self.max_tries = 1
        self.max_time = None
        self.tries = 0
        self.result = None
        self.error = False
        self.error_msg = None
        self.todo_queue_total = None
        self.calc_total = None
        self.done_queue_total = None

    def fail(self, msg=''):
        self.error = True
        self.error_msg = msg

    def succ(self, res):
        self.result = res

    def __repr__(self):
        return ('Task( proj_id = %s, task_id = %s,  module = %s, method = %s, error = %s, error_msg = %s, result = %s )' % ((self.proj_id[:8] + '...'), (self.task_id[:8] + '...'), self.module, self.method, self.error, self.error_msg, self.result))