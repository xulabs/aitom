'''
Author: Guanan Zhao
'''

import numpy as N
from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, merge, AveragePooling3D, \
    Dropout, Flatten, Activation
import keras.models as KM


# The structure defination of CNN-1 in our paper
def CNN_1(image_size, num_labels):
    num_channels = 1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels), name='input')
    
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', name='conv1')(inputs)
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', name='conv2')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='maxpool1')(m)

    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv3')(m)
    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv4')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='maxpool2')(m)

    m = Flatten(name='flatten')(m)
    m = Dense(512, activation='relu', name='fc1')(m)
    m = Dense(512, activation='relu', name='fc2')(m)
    m = Dense(num_labels, activation='linear', name='fc3')(m)
    m = Activation('softmax', name='softmax')(m)

    model = KM.Model(input=inputs, output=m)
    return model


# The structure defination of CNN-2 in our paper
def CNN_2(image_size, num_labels):
    num_channels = 1
    inputs = Input(shape = (image_size, image_size, image_size, num_channels), name='input')
    
    # the inserted layer:
    m = AveragePooling3D(pool_size=(2,2,2), strides=None, border_mode='same', name='prepooling')(inputs)
    
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', name='conv1')(m)
    m = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', name='conv2')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='maxpool1')(m)

    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv3')(m)
    m = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv4')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='maxpool2')(m)

    m = Flatten(name='flatten')(m)
    m = Dense(512, activation='relu', name='fc1')(m)
    m = Dense(512, activation='relu', name='fc2')(m)
    m = Dense(num_labels, activation='linear', name='fc3')(m)
    m = Activation('softmax', name='softmax')(m)

    model = KM.Model(input=inputs, output=m)
    return model


# The function for class prediction given a trained CNN model and the input data
def predict(model, dj):
    data = list_to_data(dj)
    pred_prob = model.predict(data['data'])      # predicted probabilities
    pred_labels = pred_prob.argmax(axis=-1)
    return pred_labels, pred_prob


# Below are pre-processing functions
def vol_to_image_stack(vs):
    image_size = vs[0].shape[0]
    sample_size = len(vs)
    num_channels=1
    sample_data = N.zeros((sample_size, image_size, image_size, image_size, num_channels), dtype=N.float32)
    for i,v in enumerate(vs):
        sample_data[i, :, :, :, 0] = v
    return sample_data

def pdb_id_label_map(pdb_ids):
    pdb_ids = set(pdb_ids)
    pdb_ids = list(pdb_ids)
    m = {p:i for i,p in enumerate(pdb_ids)}
    return m

def list_to_data(dj, pdb_id_map=None):
    re = dict()
    re['data'] = vol_to_image_stack(vs=[_['v'] for _ in dj])

    if pdb_id_map is not None:
        labels = N.array([pdb_id_map[_['pdb_id']] for _ in dj])
        from keras.utils import np_utils
        labels = np_utils.to_categorical(labels, len(pdb_id_map))
        re['labels'] = labels
    return re
