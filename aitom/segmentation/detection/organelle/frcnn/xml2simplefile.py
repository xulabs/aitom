# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:09:15 2018

@author: Berothy
"""

import numpy as np
import os
try: 
  import xml.etree.cElementTree as ET 
except ImportError: 
  import xml.etree.ElementTree as ET 
  
xmlpath = 'H:/Mitochondria/Annotations/'
smfpath = './mito_simple_label.txt'
simplefile = open(smfpath,'w')
for file in os.listdir(xmlpath):
    filepath = xmlpath + file
    tree = ET.parse(filepath)
    root = tree.getroot()
    for anno in root.iter():
        if anno.tag == 'path':
            path = anno.text       
#==============================================================================
#             if path.split('\\')[2] == 'JPEGImages':
#                 path = path.replace('JPEGImages','JPEGImages_denoise+equalize')
#==============================================================================
        if anno.tag == 'name':
            class_name = anno.text
        if anno.tag == 'xmin':
            xmin = anno.text
        if anno.tag == 'ymin':
            ymin = anno.text
        if anno.tag == 'xmax':
            xmax = anno.text
        if anno.tag == 'ymax':
            ymax = anno.text

    simplefile.write('{},{},{},{},{},{}\n'.format(path,xmin,ymin,xmax,ymax,class_name))
    
simplefile.close()