
# configs for CIFAR10

NUM_CLASS = 10
NUM_TRAIN = 50000
NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 128
SUBSET    = 25000
ADDENDUM  = 2500


TRIALS = 3
CYCLES = 7

EPOCH = 200
LR = 0.1
MILESTONES = [160]


MOMENTUM = 0.9
WDECAY = 5e-4


SCHEME = 1  # 0: expected-gradnorm scheme;   1: entropy-grandorm scheme
