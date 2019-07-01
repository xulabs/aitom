

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import os
import sys
import time
import copy
import json
import uuid
import _thread
import threading
import queue
import logging
import logging.handlers
import psutil
from aitom.tomominer.parallel.Task import Task
from aitom.tomominer.parallel.RPCServer import RPCServer


class QueueServer:

    def __init__(self):
        self.todo_queue = queue.PriorityQueue()
        self.done_queues = {}
        self.done_tasks_time = dict()
        self.done_tasks_time_max = (60.0 * 20)
        self.broadcast_todo_queue = {}
        self.start_calc = {}
        self.out_tasks = {}
        self.proj_alive_time = {}
        self.proj_alive_time_max = (60 * 5)
        self.worker_alive_time = {}
        self.worker_alive_time_max = (60 * 30)
        self.pub_logger = logging.getLogger()
        self.pub_logger.setLevel(logging.WARN)
        if False:
            h = logging.handlers.RotatingFileHandler(filename='./server.log', mode='a', maxBytes=1000000, backupCount=10)
        else:
            h = logging.handlers.RotatingFileHandler(filename='./server.log', mode='a')
        f = logging.Formatter('%(asctime)s %(host)-16s %(job_id)-16s %(source_type)-8s %(levelname)-8s %(message)s')
        h.setFormatter(f)
        self.pub_logger.addHandler(h)
        self.logger = logging.LoggerAdapter(logging.getLogger(), {'host': os.environ.get('HOSTNAME', 'unknown'), 'job_id': os.environ.get('PBS_JOBID', 'N/A').split('.')[0], 'source_type': 'queue_server', })
        self.process = psutil.Process(os.getpid())
        _thread.start_new_thread(QueueServer.remove_dead_projects_daemon, (self,))

    def log_queue_stats(self):
        self.logger.debug('TODO_QUEUE: %6d    IN_PROGRESS: %6d    DONE_QUEUE: %6d', self.todo_queue.qsize(), len(self.out_tasks), sum((_.qsize() for _ in list(self.done_queues.values()))))

    def queue_stats_string(self):
        return ('CPU %3.0f  PROJ %2d  WORKER %4d    TODO_QUEUE: %6d  IN_PROGRESS: %4d  DONE_QUEUE: %3d ' % (self.process.cpu_percent(interval=1), len(self.done_queues), self.get_worker_number(), self.todo_queue.qsize(), len(self.out_tasks), sum((_.qsize() for _ in list(self.done_queues.values())))))

    def new_project(self, proj_id, max_queue_size=0):
        lock = threading.Lock()
        with lock:
            if (proj_id not in self.done_queues):
                self.done_queues[proj_id] = queue.Queue(maxsize=max_queue_size)
        self.project_keep_alive(proj_id)
        return True

    def del_project(self, proj_id=None):
        if (proj_id in self.proj_alive_time):
            del self.proj_alive_time[proj_id]
        out_tasks_to_delete = []
        for task_id in self.out_tasks:
            if (self.out_tasks[task_id].proj_id in self.proj_alive_time):
                continue
            out_tasks_to_delete.append(task_id)
        for task_id in out_tasks_to_delete:
            try:
                del self.out_tasks[task_id]
            except:
                pass
        done_queues_to_delete = []
        for proj_id in self.done_queues:
            if (proj_id in self.proj_alive_time):
                continue
            done_queues_to_delete.append(proj_id)
        cur_time = time.time()
        for task_id in copy.deepcopy(list(self.done_tasks_time.keys())):
            if ((task_id in self.done_tasks_time) and ((cur_time - self.done_tasks_time[task_id]) > self.done_tasks_time_max)):
                try:
                    del self.done_tasks_time[task_id]
                except:
                    pass
        try:
            lock = threading.Lock()
            with lock:
                for proj_id in done_queues_to_delete:
                    del self.done_queues[proj_id]
            return True
        except:
            return False

    def project_keep_alive(self, proj_id):
        self.proj_alive_time[proj_id] = time.time()
        return True

    def get_project_alive_time(self):
        return self.proj_alive_time

    def remove_dead_projects_daemon(self, interval=60):
        while True:
            time.sleep(interval)
            self.remove_dead_projects()

    def remove_dead_projects(self):
        dead_projects = []
        for proj_id_t in self.proj_alive_time:
            time_diff = (time.time() - self.proj_alive_time[proj_id_t])
            if (time_diff > self.proj_alive_time_max):
                dead_projects.append(proj_id_t)
        for proj_id_t in dead_projects:
            del self.proj_alive_time[proj_id_t]
        self.del_project()
        return dead_projects

    def get_worker_number(self):
        cur_time = time.time()
        worker_ids = copy.copy(list(self.worker_alive_time.keys()))
        c = 0
        for _ in worker_ids:
            if (_ not in self.worker_alive_time):
                continue
            if ((cur_time - self.worker_alive_time[_]) > self.worker_alive_time_max):
                continue
            c += 1
        return c

    def task_queue_size(self):
        return self.todo_queue.qsize()

    def in_progress_task_number(self):
        return len(self.out_tasks)

    def put_tasks(self, tasks):
        print(('\r' + self.queue_stats_string()), end=' ')
        for task in tasks:
            self.todo_queue.put((task.priority, task))
            self.logger.debug('put_task %s', task)

    def put_task(self, task):
        print(('\r' + self.queue_stats_string()), end=' ')
        self.todo_queue.put((task.priority, task))
        self.logger.debug('put_task %s', task)

    def cancel_task(self, task):
        raise NotImplementedError

    def get_task(self, worker_id=None, interval=5, timeout=10):
        print(('\r' + self.queue_stats_string()), end=' ')
        self.worker_alive_time[worker_id] = time.time()
        start_time = time.time()
        while ((time.time() - start_time) < timeout):
            try:
                (priority, task) = self.todo_queue.get_nowait()
                if (task.task_id in self.done_tasks_time):
                    continue
                self.start_calc[task.task_id] = time.time()
                self.out_tasks[task.task_id] = task
                return task
            except queue.Empty:
                time.sleep(interval)
        return None

    def done_tasks_contains(self, task_id):
        return (task_id in self.done_tasks_time)

    def resubmit_undone_tasks(self, task_ids):
        task_ids_t = []
        for task_id in task_ids:
            if (task_id in self.out_tasks):
                self.put_task(self.out_tasks[task_id])
                task_ids_t.append(task_id)
        return task_ids_t

    def put_result(self, worker_id, task_id, error, error_msg, result):
        print(('\r' + self.queue_stats_string()), end=' ')
        self.worker_alive_time[worker_id] = time.time()
        if (task_id in self.done_tasks_time):
            return True
        self.done_tasks_time[task_id] = time.time()
        if (not (task_id in self.out_tasks)):
            return True
        task = self.out_tasks[task_id]
        del self.out_tasks[task_id]
        task.error = error
        task.error_msg = error_msg
        task.result = result
        task.calc_total = (time.time() - self.start_calc[task.task_id])
        del self.start_calc[task.task_id]
        if (task.proj_id not in self.done_queues):
            self.new_project(task.proj_id)
        self.done_queues[task.proj_id].put(task)
        self.logger.debug('put_result: %s', task)
        return True

    def get_results(self, proj_id):
        if (proj_id not in self.done_queues):
            self.new_project(proj_id)
        self.log_queue_stats()
        print(('\r' + self.queue_stats_string()), end=' ')
        results = []
        while True:
            try:
                task = self.done_queues[proj_id].get_nowait()
                results.append(task)
            except queue.Empty:
                if (len(results) > 0):
                    self.logger.debug('get_results: (%s)', len(results))
                else:
                    time.sleep(1)
                break
        now = time.time()
        for (task_id, start_time) in list(self.start_calc.items()):
            if (task_id in self.out_tasks):
                task = self.out_tasks[task_id]
                if (task.proj_id not in self.done_queues):
                    del self.out_tasks[task_id]
                    continue
                if ((task.max_time > 0) and ((now - start_time) > task.max_time)):
                    self.logger.error('Task %s has been running for %d! Time to resubmit', task, (now - start_time))
        return results

    def put_broadcast_task(self, task):
        print(('\r' + self.queue_stats_string()), end=' ')
        task_worker_id = {}
        for worker_id in self.broadcast_todo_queue:
            if ((time.time() - self.worker_alive_time[worker_id]) > self.worker_alive_time_max):
                continue
            t = copy.deepcopy(task)
            t.task_id_original = t.task_id
            t.task_id = str(uuid.uuid4())
            self.broadcast_todo_queue[worker_id].put((t.priority, t))
            self.logger.debug('put_task %s', t)
            task_worker_id[t.task_id] = worker_id
        return task_worker_id

    def get_broadcast_task(self, worker_id):
        self.worker_alive_time[worker_id] = time.time()
        if (worker_id not in self.broadcast_todo_queue):
            self.broadcast_todo_queue[worker_id] = queue.PriorityQueue()
            return None
        try:
            (priority, task) = self.broadcast_todo_queue[worker_id].get_nowait()
            self.out_tasks[task.task_id] = task
            self.start_calc[task.task_id] = time.time()
            return task
        except queue.Empty:
            pass
        return None

    def log(self, record):
        self.pub_logger.handle(record)