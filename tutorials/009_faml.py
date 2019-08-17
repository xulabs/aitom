import os, uuid, pickle                                      
import numpy as N                       
from numpy.fft import fftn, ifftn, fftshift, ifftshift

from aitom.io.db.lsm_db import LSM
from aitom.image.vol.wedge.util import wedge_mask                                        

def fourier_transform(v):
    return fftshift(fftn(v))                        

                                  
image_db = LSM('aitom_demo_subtomograms.db') # Build a database file

with open('aitom_demo_subtomograms.pickle') as f:     subtomograms = pickle.load(f) # Load demo subtomograms pickle file

dj = [] # Key file for the database
                                                   
m = wedge_mask([32,32,32], ang1 = 30, sphere_mask=True, verbose=False) # Subtomogram missing wedge masks             

for s in subtomograms['5T2C_data']:                                                                                                                                                  
    v = fourier_transform(s)                  # Save fourier transformed subtomogram data                           
    s1 = str(uuid.uuid4())                         
    s2 = str(uuid.uuid4())       
    image_db[s1] = v                          # Save a subtomogram to the database according to a key      
    image_db[s2] = m                          # Save a subtomogram mask to the database according to a key
              
    dj.append({'v': s1, 'm': s2, 'id': '5T2C'})
    
for s in subtomograms['1KP8_data']:                                                                                                                                                  
    v = fourier_transform(s)                                             
    s1 = str(uuid.uuid4())                         
    s2 = str(uuid.uuid4())       
    image_db[s1] = v                                
    image_db[s2] = m
              
    dj.append({'v': s1, 'm': s2, 'id': '1KP8'})

with open('dj.pickle', 'wb') as f:    pickle.dump(dj, f, protocol=-1) # Save the key file
