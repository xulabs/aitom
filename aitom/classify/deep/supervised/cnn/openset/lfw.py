"""
Helper for evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from . import utils


def evaluate(embeddings, actual_issame, threshold, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 10, 0.01 / 4)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = utils.calculate_roc(thresholds,
                                             embeddings1,
                                             embeddings2,
                                             np.asarray(actual_issame),
                                             nrof_folds=nrof_folds)
    thresholds = np.arange(0, 10, 0.001)
    val, val_std, far = utils.calculate_val(thresholds,
                                            embeddings1,
                                            embeddings2,
                                            np.asarray(actual_issame),
                                            threshold,
                                            nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    file_ext = 'npy'
    for pair in pairs:
        if len(pair) == 3:
            # pair[0] + '_' + '%04d' %
            path0 = os.path.join(lfw_dir, pair[0], pair[1] + '.' + file_ext)
            # pair[0] + '_' + '%04d' %
            path1 = os.path.join(lfw_dir, pair[0], pair[2] + '.' + file_ext)
            issame = True
        if len(pair) == 4:
            # pair[0] + '_' + '%04d' %
            path0 = os.path.join(lfw_dir, pair[0], pair[1] + '.' + file_ext)
            # pair[2] + '_' + '%04d' %
            path1 = os.path.join(lfw_dir, pair[2], pair[3] + '.' + file_ext)
            issame = False
        # Only add the pair if both paths exist
        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    return path_list, issame_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    import random
    random.shuffle(pairs)
    return np.array(pairs)
