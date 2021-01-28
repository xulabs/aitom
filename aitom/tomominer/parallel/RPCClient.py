

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import socket
import pickle
import time


class RPCClient(object):

    def __init__(self, host, port, tcp_keepidle=(60 * 5), tcp_keepintvl=30, tcp_keepcnt=5):
        self.host = host
        self.port = port
        self.tcp_keepidle = tcp_keepidle
        self.tcp_keepintvl = tcp_keepintvl
        self.tcp_keepcnt = tcp_keepcnt
        self._connect()

    def __del__(self):
        self._close()

    def _connect(self, timeout=None, delay=10, max_tries=0):
        n_tries = 0
        while True:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.tcp_keepidle)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.tcp_keepcnt)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.tcp_keepintvl)
            try:
                self.socket.connect((self.host, self.port))
                self.rfile = self.socket.makefile('rb')
                self.wfile = self.socket.makefile('wb')
                return
            except socket.error as exc:
                raise
                try:
                    self.socket.shotdown(socket.SHUT_RDWR)
                    self.socket.close()
                except:
                    continue
                n_tries += 1
                if (n_tries >= max_tries):
                    raise
                time.sleep(delay)
                continue

    def _close(self):
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        self.rfile.close()
        self.wfile.close()

    def __getattr__(self, name):
        if name.startswith('_'):
            print('Warning:', ("Cannot call method (%s) with leading '_'" % name))
            return

        def proxy(*args, **kwargs):
            while True:
                try:
                    pickle.dump({'method': name, 'args': args, 'kwargs': kwargs, }, self.wfile, protocol=2)
                    self.wfile.flush()
                    (status, result) = pickle.load(self.rfile)
                    if (status == 'OK'):
                        return result
                    else:
                        raise result
                except socket.timeout:
                    time.sleep(5.0)
                    continue
                except (socket.error, EOFError) as e:
                    raise
                    try:
                        self._close()
                    except:
                        pass
                    self._connect()
        return proxy