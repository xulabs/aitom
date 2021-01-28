

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import time
import os
import sys
import uuid
import pickle
import _thread
import numpy as N
from aitom.tomominer.parallel.RPCClient import RPCClient
from aitom.tomominer.parallel.RPCLoggingHandler import RPCLoggingHandler
from aitom.tomominer.parallel.Task import Task
import logging


class QueueMaster:

    def __init__(self, host, port):
        self.work_queue = RPCClient(host, port)
        self.handler = RPCLoggingHandler(self.work_queue)
        self.logger = logging.getLogger()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger = logging.LoggerAdapter(logging.getLogger(), {'host': os.environ.get('HOSTNAME', 'unknown'), 'job_id': os.environ.get('PBS_JOBID', 'N/A').split('.')[0], 'source_type': 'queue_master', })
        self.proj_id = str(uuid.uuid4())
        self.work_queue.new_project(self.proj_id)
        _thread.start_new_thread(QueueMaster.keep_alive, (self, RPCClient(host, port)))

    def task(self, priority=1000, module=None, method=None, args=[], kwargs={}):
        return Task(priority=priority, proj_id=self.proj_id, module=module, method=method, args=args, kwargs=kwargs)

    def __del__(self):
        self.work_queue.del_project(self.proj_id)
        return

    def keep_alive(self, work_queue_alive, interval=10):
        while True:
            work_queue_alive.project_keep_alive(self.proj_id)
            time.sleep(interval)

    def run(self, tasks, max_time=None, max_retry=1000, one_at_a_time=False):
        task_dict = {}
        state = {}
        results = {}
        for t in tasks:
            t.max_time = max_time
            task_dict[t.task_id] = t
            state[t.task_id] = max_retry
        if one_at_a_time:
            for t in tasks:
                self.work_queue.put_task(t)
        else:
            self.work_queue.put_tasks(tasks)
        while len(state):
            results = self.work_queue.get_results(self.proj_id)
            for res in results:
                if (res.task_id not in state):
                    self.logger.warning('recieved result from an unknown task: %s', (res,))
                    continue
                if res.error:
                    self.logger.debug('result %s, contains error flag.  task raised exception remotely: %s', res.task_id, res)
                    state[res.task_id] -= 1
                    if (state[res.task_id] > 0):
                        self.logger.warning('resubmitting crashed task: %s', res.task_id)
                        self.work_queue.put_task(task_dict[res.task_id])
                        continue
                    else:
                        self.logger.warning('task failed too many times: %s', res.task_id)
                del state[res.task_id]
                yield res
            if ((self.work_queue.task_queue_size() == 0) and (len(state) > 0)):
                undone_tasks = []
                for task_id in list(state.keys()):
                    if (state[task_id] <= 0):
                        continue
                    undone_tasks.append(task_id)
                if (len(undone_tasks) > 0):
                    re_submitted_tasks = self.work_queue.resubmit_undone_tasks(undone_tasks)
                    if (len(re_submitted_tasks) > 0):
                        self.logger.warning('resubmitted in progress %d  tasks', len(re_submitted_tasks))
                        for task_id in re_submitted_tasks:
                            state[task_id] -= 1

    def run__except(self, tasks, one_at_a_time=False):
        task_num = float(len(tasks))
        count = 0
        for res in self.run(tasks, one_at_a_time=one_at_a_time):
            count += 1
            sys.stdout.write(('%d   %0.3f    \r' % (count, (count / task_num))))
            sys.stdout.flush()
            if res.error:
                print('Computation Failed!')
                print(res)
                print('task_id      :', res.task_id)
                print('method       :', res.method)
                print('args         :', res.args)
                print('kwargs       :', res.kwargs)
                print('error_msg    :', res.error_msg)
                raise Exception
            yield res

    def estimate_chunk_size(self, n, worker_number_multiply_factor=1.5):
        n_chunk = (float(n) / (self.work_queue.get_worker_number() * float(worker_number_multiply_factor)))
        n_chunk = N.max((n_chunk, 1))
        n_chunk = int(N.ceil(n_chunk))
        return n_chunk

    def broadcast(self, task):
        task_dict = {}
        state = {}
        results = {}
        state = self.work_queue.put_broadcast_task(task)
        return state

    def broadcast_collect(self, state, timeout=120):
        cur_time = time.time()
        while len(state):
            results = self.work_queue.get_results(self.proj_id)
            for res in results:
                if (res.task_id not in state):
                    self.logger.warning('recieved result from an unknown task: %s', (res,))
                    continue
                if res.error:
                    self.logger.debug('result %s, contains error flag.  task raised exception remotely: %s', res.task_id, res)
                del state[res.task_id]
                yield res
            if ((time.time() - cur_time) > timeout):
                break

    def run_broadcast__except(self, task):
        state = self.broadcast(task)
        task_num = float(len(state))
        count = 0
        for res in self.broadcast_collect(state):
            count += 1
            sys.stdout.write(('%d   %0.3f    \r' % (count, (count / task_num))))
            sys.stdout.flush()
            if res.error:
                print('Computation Failed!')
                print(res)
                print('task_id      :', res.task_id)
                print('method       :', res.method)
                print('args         :', res.args)
                print('kwargs       :', res.kwargs)
                print('error_msg    :', res.error_msg)
                raise Exception
            yield res