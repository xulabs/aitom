import torch
import torch.nn as nn
import math
import numpy as np
import os
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnns
import time
import cv2
from datetime import datetime


class SubtomogramNShotDataset(Dataset):
    def __init__(self, data_root, batch_size, class_per_set, sample_per_class):
        print('loading {}'.format(data_root))

        self.cnt = np.load(os.path.join(data_root, 'cnt.npy'))
        data = np.load(os.path.join(data_root, 'data.npy'))
        data_seg = np.load(os.path.join(data_root, 'data_seg.npy'))

        data = np.expand_dims(data, axis=5)
        data_seg = np.expand_dims(data_seg, axis=5)

        self.train_toms = data[:14]
        self.train_toms_seg = data_seg[:14]
        self.train_tom_cnt = self.cnt[:14]
        self.test_toms = data[14:]
        self.test_toms_seg = data_seg[14:]
        self.test_tom_cnt = self.cnt[14:]

        self.batch_size = batch_size
        self.n_classes = data.shape[0]
        self.class_per_set = class_per_set
        self.sample_per_class = sample_per_class

    def __len__(self):
        return sum(self.cnt)

    def sample_new_train_batch(self):
        x1 = np.zeros((self.batch_size, self.train_toms.shape[2], self.train_toms.shape[3], self.train_toms.shape[4], self.train_toms.shape[5]), np.float32)
        x1_seg = np.zeros((self.batch_size, self.train_toms.shape[2], self.train_toms.shape[3], self.train_toms.shape[4], self.train_toms.shape[5]), np.float32)
        y1 = np.zeros((self.batch_size, 1), np.int32)
        for i in range(self.batch_size):
            rclass = np.random.randint(self.train_toms.shape[0])
            rindex = np.random.randint(self.train_tom_cnt[rclass])
            x1[i] = self.train_toms[rclass][rindex]
            x1_seg[i] = self.train_toms_seg[rclass][rindex]
            y1[i] = rclass

        return x1, x1_seg, y1

    def sample_new_test_batch(self):
        x1 = np.zeros((self.batch_size, self.test_toms.shape[2], self.test_toms.shape[3], self.test_toms.shape[4], self.test_toms.shape[5]), np.float32)
        x1_seg = np.zeros((self.batch_size, self.test_toms.shape[2], self.test_toms.shape[3], self.test_toms.shape[4], self.test_toms.shape[5]), np.float32)
        y1 = np.zeros((self.batch_size, 1), np.int32)

        choose_classes = np.random.choice(self.test_toms.shape[0], size=self.class_per_set, replace=False)
        x2 = np.zeros((self.class_per_set, self.batch_size, self.test_toms.shape[2], self.test_toms.shape[3], self.test_toms.shape[4], self.test_toms.shape[5]), np.float32)
        x2_seg = np.zeros((self.class_per_set, self.batch_size, self.test_toms.shape[2], self.test_toms.shape[3], self.test_toms.shape[4], self.test_toms.shape[5]), np.float32)
        y2 = np.zeros((self.class_per_set, self.batch_size, 1), np.int32)

        for i in range(self.batch_size):
            rclass = np.random.randint(self.class_per_set)
            rindex = np.random.randint(self.test_tom_cnt[choose_classes[rclass]])
            x1[i] = self.test_toms[choose_classes[rclass]][rindex]
            x1_seg[i] = self.test_toms_seg[choose_classes[rclass]][rindex]
            y1[i] = choose_classes[rclass]

        for j in range(self.class_per_set):
            for i in range(self.batch_size):
                rindex = np.random.randint(self.test_tom_cnt[rclass])
                x2[j][i] = self.test_toms[choose_classes[j]][rindex]
                x2_seg[j][i] = self.test_toms_seg[choose_classes[j]][rindex]
                y2[j][i] = choose_classes[j]

        return x1, x1_seg, y1, x2, x2_seg, y2

