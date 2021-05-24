# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:42:15 2018

@author: Berothy
"""

import cv2
import numpy as np

def data_split(input_path,train_path,test_path):    
    with open(input_path, 'r') as f:
        tomos = {}
        print('Parsing annotation files')
        train_num = 0
        test_num = 0
        train = open(train_path,'w')
        test = open(test_path,'w')
        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split
            #generating label file for each tomogram as 'NAME_OF_TOMOGRAM_label.txt'
            tomo = '_'.join(filename.split('_')[:-1])
            if tomo not in tomos:
                tomos[tomo]='./{}_label.txt'.format(tomo.split('\\')[3])
                tomo_file = open(tomos[tomo],'w')
            tomo_file.write('{},{},{},{},{},{}\n'.format(filename,x1,y1,x2,y2,class_name))
            #generating label file for training and testing
            if np.random.randint(0, 6) > 0:
                train_num += 1
                train.write('{},{},{},{},{},{}\n'.format(filename,x1,y1,x2,y2,class_name))
            else:
                test_num += 1
                test.write('{},{},{},{},{},{}\n'.format(filename,x1,y1,x2,y2,class_name))
        print('train set:{}\ntest set:{}\n'.format(train_num,test_num))
                    
                
input_path = './mito_simple_label.txt'
train_path = './mito_train_label.txt'
test_path = './mito_test_label.txt'
data_split(input_path,train_path,test_path)