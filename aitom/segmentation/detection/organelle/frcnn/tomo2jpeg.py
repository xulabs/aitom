#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:32:30 2018

@author: mac
"""

from EMAN2 import *
import numpy as np
import os
import cv2
import gc

path = "H:/Mitochondria_tomogram"
dirs = os.listdir(path)
outpath = "./original_images"

if not os.path.exists(outpath):
    os.makedirs(outpath)

for file in dirs:
    filepath = path + '/' + file
    if os.path.splitext(filepath)[1] == '.mrc' or os.path.splitext(filepath)[1] == '.st':
        data = EMData(filepath)
        data_array = EMNumPy.em2numpy(data)
        for i in range(data_array.shape[2]):
            img = data_array[:,:,i]
            imgname = outpath + '/' + file + '_' + str(i) + '.jpg'
            cv2.imwrite(imgname,img)
            print(os.path.splitext(file)[0]+'_'+str(i)+'saved as jpeg!')
        del data
        gc.collect()

            
