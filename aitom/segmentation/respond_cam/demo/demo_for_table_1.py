'''
Author: Guanan Zhao
'''

# Table 1: Comparison between Grad-CAM and Respond-CAM on L1 error and
# Kendall's Tau (KT) on different CNNs and datasets.

import sys
sys.path.append('../')

import os
output_dir = './output/demo_for_table_1'
try: # If output_dir does not exist, we make it
    os.makedirs(output_dir)
except: # Otherwise, do nothing
    pass

from keras.models import load_model
cnn_1_nfree = load_model('../data/nfree_CNN_1.h5')
cnn_2_nfree = load_model('../data/nfree_CNN_2.h5')
cnn_1_noisy = load_model('../data/noisy_CNN_1.h5')
cnn_2_noisy = load_model('../data/noisy_CNN_2.h5')

import pickle
with open('../data/test_set_nfree.pickle') as f:
    dj_nfree = pickle.load(f)
with open('../data/test_set_noisy.pickle') as f:
    dj_noisy = pickle.load(f)


import respond_cam as R
import numpy as np

def L1(camsum, score):
    return np.mean(np.abs(camsum - score))

def KT(camsum, score):
    from scipy.stats import kendalltau
    tau, p_value = kendalltau(camsum, score)
    return tau

def mean_std(some_list):
    array = np.array(some_list)
    return np.mean(array), np.std(array)

configs = [
  {'cnn': cnn_1_nfree, 'dj': dj_nfree, 'msg': 'CNN-1, SNR=inf'},
  {'cnn': cnn_2_nfree, 'dj': dj_nfree, 'msg': 'CNN-2, SNR=inf'},
  {'cnn': cnn_1_noisy, 'dj': dj_noisy, 'msg': 'CNN-1, SNR=0.1'},
  {'cnn': cnn_2_noisy, 'dj': dj_noisy, 'msg': 'CNN-2, SNR=0.1'}
]

lines = []
for config in configs:
    # Below generating the raw data.
    print('Now evaluating on:', config['msg'])
    lines.append(config['msg'] + '\n')
    cnn = config['cnn']
    dj = config['dj']
    camsums_grad, camsums_respond, scores = R.get_all_scores_and_camsums(cnn, 'maxpool2', dj)

    # Below calculating the value for Table 1.
    L1_grad = mean_std([L1(c, s) for c, s in zip(camsums_grad, scores)])
    L1_respond = mean_std([L1(c, s) for c, s in zip(camsums_respond, scores)])
    KT_grad = mean_std([KT(c, s) for c, s in zip(camsums_grad, scores)])
    KT_respond = mean_std([KT(c, s) for c, s in zip(camsums_respond, scores)])

    for name, stat in zip(('L1, Grad', 'L1, Respond', 'KT, Grad', 'KT, Respond'), 
      (L1_grad, L1_respond, KT_grad, KT_respond)):
        print '%s    Mean=%.3f  Std=%.3f' % (name, stat[0], stat[1])
        lines.append('%s    Mean=%.3f  Std=%.3f\n' % (name, stat[0], stat[1]))
    print
    lines.append('\n')

with open(os.path.join(output_dir, 'stats.txt'), 'w') as f:
    f.writelines(lines)
