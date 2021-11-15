"""
Prelimary code extracted from the code developed for our following paper
Xu M, Chai X, Muthakana H, Liang X, Yang G, Zeev-Ben-Mordehai T, Xing E. Deep learning based
subdivision approach for large scale macromolecules structure recovery from electron cryo
tomograms. Preprint: arXiv:1701.08404.  ISMB 2017 (acceptance rate 16%),
Bioinformatics doi:10.1093/bioinformatics/btx230

The current version of this code is mainly for experienced programmers for inspection purpose.
It is subject for further updatess to be made more friendly to end users.
Please cite the paper when this code is used or adapted for your research publication.

License: ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
"""

import numpy as N

from keras.layers import Input, Dense, Conv3D, MaxPooling3D,\
    merge, ZeroPadding3D, AveragePooling3D, Dropout, Flatten, Activation
import keras.models as KM


def inception3D(image_size, num_labels):
    num_channels = 1
    inputs = Input(shape=(image_size, image_size, image_size, num_channels))

    m = Conv3D(32, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='relu',
                      padding='valid', input_shape=())(inputs)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='same')(m)

    # inception module 0
    branch1x1 = Conv3D(32, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu',
                              padding='same')(m)
    branch3x3_reduce = Conv3D(32, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu',
                                     padding='same')(m)
    branch3x3 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu',
                              padding='same')(branch3x3_reduce)
    branch5x5_reduce = Conv3D(16, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu',
                                     padding='same')(m)
    branch5x5 = Conv3D(32, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='relu',
                              padding='same')(branch5x5_reduce)
    branch_pool = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(m)
    branch_pool_proj = Conv3D(32, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu',
                                     padding='same')(branch_pool)
    # m = merge([branch1x1, branch3x3, branch5x5, branch_pool_proj], mode='concat', concat_axis=-1)
    from keras.layers import concatenate
    m = concatenate([branch1x1, branch3x3, branch5x5, branch_pool_proj], axis=-1)

    m = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid')(m)
    m = Flatten()(m)
    m = Dropout(0.7)(m)

    # expliciately seperate Dense and Activation layers in order for projecting to structural feature space
    m = Dense(num_labels, activation='linear')(m)
    m = Activation('softmax')(m)

    mod = KM.Model(inputs=inputs, outputs=m)

    return mod


def dsrff3D(image_size, num_labels):
    num_channels = 1
    inputs = Input(shape=(image_size, image_size, image_size, num_channels))

    # modified VGG19 architecture
    bn_axis = 3
    m = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    m = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(m)
    m = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Flatten(name='flatten')(m)
    m = Dense(512, activation='relu', name='fc1')(m)
    m = Dense(512, activation='relu', name='fc2')(m)
    m = Dense(num_labels, activation='softmax')(m)

    mod = KM.Model(inputs=inputs, outputs=m)

    return mod


def compile(model, num_gpus=1):
    if num_gpus > 1:
        import keras_extras.utils.multi_gpu as KUM
        model = KUM.make_parallel(model, num_gpus)

    import tensorflow.keras.optimizers as KOP
    kop = KOP.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=kop, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train(model, dj, pdb_id_map, nb_epoch):
    dl = list_to_data(dj, pdb_id_map)

    model.fit(dl['data'], dl['labels'], nb_epoch=nb_epoch, shuffle=True,
              validation_split=validation_split)


def train_validation(model, dj, pdb_id_map, nb_epoch, validation_split):
    from sklearn.model_selection import train_test_split
    sp = train_test_split(dj, test_size=validation_split)
    train_dl = list_to_data(sp[0], pdb_id_map)
    validation_dl = list_to_data(sp[1], pdb_id_map)

    model.fit(train_dl['data'], train_dl['labels'],
              validation_data=(validation_dl['data'], validation_dl['labels']),
              epochs=nb_epoch, shuffle=True)


def predict(model, dj):
    data = list_to_data(dj)
    # predicted probabilities
    pred_prob = model.predict(data)
    pred_labels = pred_prob.argmax(axis=-1)

    return pred_labels


def vol_to_image_stack(vs):
    image_size = vs[0].shape[0]
    sample_size = len(vs)
    num_channels = 1

    sample_data = N.zeros((sample_size, image_size, image_size, image_size, num_channels),
                          dtype=N.float32)

    for i, v in enumerate(vs):
        sample_data[i, :, :, :, 0] = v

    return sample_data


def pdb_id_label_map(pdb_ids):
    pdb_ids = set(pdb_ids)
    pdb_ids = list(pdb_ids)
    m = {p: i for i, p in enumerate(pdb_ids)}
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


def convert_demo_data_format(dj, label_list):
    """
    demo dataset is a dict,key is the label,
    value is a matrix size of (number * size * size * size)
        {'label1':[[[]]],
         'label2':[[[]]]}

    convert it to a list of dict
        [
            {'pdb_id':'label1',
            'v':[[[]]]},
            {'pdb_id':'label2',
            'v':[[[]]]},
            {'pdb_id':'label3',
            'v':[[[]]]}
        ]
    """
    re = []
    for every_lable in label_list:
        for every_pic in dj[every_lable]:
            re.append({'pdb_id': every_lable, 'v': every_pic})

    import random
    random.shuffle(re)
    return re


if __name__ == '__main__':
    import pickle
    with open('./aitom_demo_subtomograms.pickle', 'rb') as f:
        dj0 = pickle.load(f, encoding='iso-8859-1')

    label_list = ['5T2C_data', '1KP8_data']

    dj = convert_demo_data_format(dj0, label_list)
    pdb_id_map = pdb_id_label_map([_['pdb_id'] for _ in dj])

    model = inception3D(image_size=dj[0]['v'].shape[0], num_labels=len(pdb_id_map))
    # model = dsrff3D(image_size=dj[0]['v'].shape[0], num_labels=len(pdb_id_map))
    model = compile(model)
    train_validation(model=model, dj=dj, pdb_id_map=pdb_id_map, nb_epoch=10, validation_split=0.2)
