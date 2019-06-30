

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import socketserver
import pickle
import time
import sys
import select
import os
import psutil


class RPCHandler(socketserver.StreamRequestHandler):

    def handle(self):
        self.server.active_connections += 1
        while True:
            try:
                data = pickle.load(self.rfile)
            except EOFError:
                break
            try:
                result = self.server._dispatch(data)
            except Exception as e:
                pickle.dump(('ERR', e), self.wfile, protocol=2)
                if True:
                    import traceback
                    traceback.print_exc()
            else:
                pickle.dump(('OK', result), self.wfile, protocol=2)
            self.wfile.flush()
        self.server.active_connections -= 1


class RPCServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    active_connections = 0

    def __init__(self, addr, requestHandler=RPCHandler, bind_and_activate=True):
        self.instance = None
        socketserver.ThreadingTCPServer.__init__(self, addr, requestHandler, bind_and_activate)
        self.previous_method = None
        self.same_method_call_count = 0
        self.process = psutil.Process(os.getpid())

    def register_instance(self, obj):
        self.instance = obj

    def _dispatch(self, data, cpu_usage_threshold=200):
        try:
            method = data['method']
            args = data['args']
            kwargs = data['kwargs']
        except:
            raise
        if (self.instance is None):
            raise Exception('No instance installed on the server.')
        if False:
            if (method != self.previous_method):
                self.previous_method = method
                self.same_method_call_count = 0
                sys.stdout.write('\n')
            self.same_method_call_count += 1
            sys.stdout.write((((('\r' + method) + ' ') + repr(self.same_method_call_count)) + '\t'))
        while True:
            cpu_usage = self.process.cpu_percent(interval=1)
            if (cpu_usage < cpu_usage_threshold):
                break
            time.sleep(1)
        if method.startswith('_'):
            raise AttributeError(("Cannot call method (%s) with leading '_'" % method))
        if hasattr(self.instance, method):
            func = getattr(self.instance, method)
            if (not callable(func)):
                raise AttributeError(('Requested function (%s) is not callable' % method))
            return func(*args, **kwargs)
        else:
            raise AttributeError('Requested function (%s) not found in instance', method)

    def handle_error(self, request, client_address):
        return
        print(('-' * 40))
        print('Exception happened during processing of request from', end=' ')
        print(client_address)
        import traceback
        traceback.print_exc()
        print(('-' * 40))