

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import logging


class RPCLoggingHandler(logging.Handler):

    def __init__(self, rpc):
        logging.Handler.__init__(self)
        self.rpc = rpc

    def emit(self, record):
        try:
            ei = record.exc_info
            if ei:
                dummy = self.format(record)
                record.exc_info = None
            self.rpc.log(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)