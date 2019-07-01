

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import traceback
import logging
import os
import sys
import time
import warnings
import uuid
from multiprocessing.pool import Pool
import importlib
from aitom.tomominer.parallel.RPCClient import RPCClient
from aitom.tomominer.parallel.RPCLoggingHandler import RPCLoggingHandler
from aitom.tomominer.parallel.Task import Task
from aitom.tomominer.io.cache import Cache


class QueueWorker:

    def __init__(self, host=None, port=None, instance=None, pool=None, tmp_dir=None):
        self.worker_id = str(uuid.uuid4())
        self.work_queue = RPCClient(host, port)
        self.handler = RPCLoggingHandler(self.work_queue)
        self.logger = logging.getLogger()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger = logging.LoggerAdapter(logging.getLogger(), {'host': os.environ.get('HOSTNAME', 'unknown'), 'job_id': os.environ.get('PBS_JOBID', 'N/A').split('.')[0], 'source_type': 'queue_worker', })
        if (tmp_dir is None):
            tmp_dir = os.getenv('TOMOMINER_TMP_DIR')
        assert (tmp_dir is not None)
        self.cache = Cache(tmp_dir=tmp_dir, logger=self.logger)
        self.cache_none = Cache(logger=self.logger)
        self.pool = pool

    def run(self, interval=5):
        while True:
            task = self.work_queue.get_broadcast_task(worker_id=self.worker_id)
            if (not task):
                task = self.work_queue.get_task(worker_id=self.worker_id)
            if (not task):
                time.sleep(interval)
                continue
            self.task = task
            (err, err_msg, result) = self._dispatch()
            if err:
                self.logger.warning('task failed: %s, %s', repr(task), err_msg)
                continue
            while True:
                if self.work_queue.done_tasks_contains(task.task_id):
                    break
                if self.work_queue.put_result(worker_id=self.worker_id, task_id=task.task_id, error=err, error_msg=err_msg, result=result):
                    break
                time.sleep(10)

    def _dispatch(self):
        if self.task.method.startswith('_'):
            return (True, "method starts with '_'", None)
        else:
            assert (self.task.module is not None)
            try:
                modu = importlib.import_module(self.task.module)
            except Exception:
                (ex_type, ex, tb) = sys.exc_info()
                return (True, ('loading module error: %s  in sys path  %s     ;    exception  %s' % (self.task.module, repr(sys.path), repr(traceback.format_tb(tb)))), None)
            try:
                func = getattr(modu, self.task.method)
            except:
                return (True, ('method not found: %s ' % self.task.method), None)
            if (not callable(func)):
                return (True, 'method not callable', None)
            try:
                assert ('self' not in self.task.kwargs)
                self.task.kwargs['self'] = self
                result = func(*self.task.args, **self.task.kwargs)
                return (False, None, result)
            except Exception as ex:
                return (True, traceback.format_exc(), None)
                sys.stderr.write(traceback.format_exc())
                self.logger.error('Exception: %s', traceback.format_exc())