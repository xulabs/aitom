"""
utility functions for multi processing
"""

import importlib
import multiprocessing
import sys
import time
from multiprocessing.pool import Pool


def run_iterator(tasks, worker_num=multiprocessing.cpu_count(), verbose=False):
    """
    given a list of tasks, using multiprocessing to run and collect results
    if worker_num <= 1, just go for single processing
    """
    if verbose:
        print('tomominer.parallel.multiprocessing.util.run_iterator()', 'start', time.time())

    worker_num = min(worker_num, multiprocessing.cpu_count())

    for i, t in tasks.items():
        if 'args' not in t:
            t['args'] = ()
        if 'kwargs' not in t:
            t['kwargs'] = {}
        if 'id' not in t:
            t['id'] = i
        assert t['id'] == i

    completed_count = 0
    if worker_num > 1:
        pool = Pool(processes=worker_num)
        pool_apply = []
        for i, t in tasks.items():
            # aa = pool.apply_async(func=t['func'], args=t['args'], kwds=t['kwargs'])
            aa = pool.apply_async(func=call_func, kwds={'t': t})

            pool_apply.append(aa)

        for pa in pool_apply:
            yield pa.get(99999)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()

        pool.close()
        pool.join()
        del pool

    else:
        for i, t in tasks.items():
            yield call_func(t)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()

    if verbose:
        print('tomominer.parallel.multiprocessing.util.run_iterator()', 'end', time.time())


# alias
run_batch = run_iterator


def call_func(t):
    '''
    Generate random seeds for numpy, so that different cores have different random seeds, see
    https://discuss.pytorch.org/t/why-does-numpy-random-rand-produce-the-same-values-in-different-cores/12005
    '''
    import numpy as N
    N.random.seed(random.randint(0,123456789))
    
    # call the function
    if 'func' in t:
        assert 'module' not in t
        assert 'method' not in t
        func = t['func']
    else:
        modu = importlib.import_module(t['module'])
        func = getattr(modu, t['method'])

    r = func(*t['args'], **t['kwargs'])
    return {'id': t['id'], 'result': r}


def run_batch_test__foo(a, b, c=0):
    return a + b + c


def run_batch_test():
    ts = {}
    for i in range(100):
        ts[i] = {'func': run_batch_test__foo, 'args': (i, i + 1), 'kwargs': {'c': i + 2}}

    rs = run_batch(ts, worker_num=2)
    print([_ for _ in rs])


"""
#ipython --pdb

%reset -f
import aitom.parallel.multiprocessing.util as TPMU;         reload(TPMU);
TPMU.run_batch_test()
"""
