# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 22:51:04 2018

@author: Berothy
"""

import cv2
import os

path = './original_images/'
outpath = './preprocessed_images/'
for file in os.listdir(path):
    imgpath = path + file
    img = cv2.imread(imgpath,0)
    blur = cv2.bilateralFilter(img,9,100,1.2)
    equ = cv2.equalizeHist(blur)
    out = outpath+file
    cv2.imwrite(out,equ)