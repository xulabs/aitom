import os
import uuid
import pickle
import numpy as N
from numpy.fft import fftn, ifftn, fftshift, ifftshift

from aitom.io.db.lsm_db import LSM
from aitom.image.vol.wedge.util import wedge_mask


def fourier_transform(v):
    return fftshift(fftn(v))


image_db = LSM('data/aitom_demo_subtomograms.db')  # Build a database file

with open('data/aitom_demo_subtomograms.pickle', 'rb') as f:
    # Load demo subtomograms pickle file
    subtomograms = pickle.load(f, encoding='iso-8859-1')

dj = []  # Key file for the database

m = wedge_mask([32, 32, 32], ang1=30, sphere_mask=True,
               verbose=False)  # Subtomogram missing wedge masks

for s in subtomograms['5T2C_data']:
    # Save fourier transformed subtomogram data
    v = fourier_transform(s)
    s1 = str(uuid.uuid4())
    s2 = str(uuid.uuid4())
    # Save a subtomogram to the database according to a key
    image_db[s1] = v
    # Save a subtomogram mask to the database according to a key
    image_db[s2] = m

    dj.append({'v': s1, 'm': s2, 'id': '5T2C'})

for s in subtomograms['1KP8_data']:
    v = fourier_transform(s)
    s1 = str(uuid.uuid4())
    s2 = str(uuid.uuid4())
    image_db[s1] = v
    image_db[s2] = m

    dj.append({'v': s1, 'm': s2, 'id': '1KP8'})

with open('data/dj.pickle', 'wb') as f:
    pickle.dump(dj, f, protocol=-1)  # Save the key file

img_data = {}
img_data['db_path'] = 'data/aitom_demo_subtomograms.db'
with open('data/dj.pickle', 'rb') as f:
    img_data['dj'] = pickle.load(f)

import aitom.average.ml.faml.faml as faml
faml.test_EM_real_data(img_data, 20, 2, 5, "output")

