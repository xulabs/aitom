from __future__ import division
import numpy as np
import math, pickle, os, pdb
from .model import FCN8, FCN1, FCN_aspp, FCN_ed, FCN_ed2
import tensorflow as tf

import keras
from keras import optimizers, metrics
from keras.models import Sequential, Model, load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K


def loader(snr):
    path = 'dataset/'

    dataset = np.load(os.path.join(path, 'dataset_snr' + str(snr) + '.npz'))
    data = dataset['data']
    label = dataset['label']

    data = np.expand_dims(data, axis=4)
    label1 = np.reshape(label, (label.shape[0], (np.prod(label.shape[1:]))))
    label = keras.utils.to_categorical(label1, num_classes=2)

    num = data.shape[0]
    tmp = int(0.9 * num)
    # Data preprocessing
    trainX = data[0:tmp]
    trainY = label1[0:tmp]
    trainY = keras.utils.to_categorical(trainY, num_classes=2)

    testX = data[tmp:]
    testY = label1[tmp:]
    testY = keras.utils.to_categorical(testY, num_classes=2)
    return trainX, trainY, testX, testY


def iou(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.mean_iou(y_true, y_pred, 2)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables()]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def meanIoU(y_pred, y_true):
    iou = np.zeros(2)
    y_pred = np.argmax(y_pred, axis=-1).astype(bool)
    y_true = np.argmax(y_true, axis=-1).astype(bool)

    al = y_pred.shape[1]
    pos = np.sum(y_pred * y_true, axis=1)
    neg = np.sum((~y_pred) * (~y_true), axis=1)
    # pos=float(np.sum(y_pred * y_true))
    # neg=float(np.sum((~y_pred) * (~y_true)))

    iou[0] = np.mean(neg / (al - pos))
    iou[1] = np.mean(pos / (al - neg))

    return np.mean(iou)


# (NONE, 40^3, 2) -->(NONE,40,40,40)
def prediction_reshape(prediction):
    y_pred = np.argmax(prediction, axis=-1)
    out = np.reshape(y_pred, (y_pred.shape[0], 40, 40, 40))
    return out


if __name__ == '__main__':
    name = 'v2FCN_ed_snr100'

trainX, trainY, testX, testY = loader(500)  # arg: snr= 10000, 500, 100
# model=FCN8(data.shape[1:])  model=FCN_aspp(trainX.shape[1:])
batch = 128
model = FCN_ed(trainX.shape[1:])
model.summary()
# Compile
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=1e-8)
# sgd=optimizers.SGD(lr=0.005, momentum=0.9, decay=0, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

path1 = os.path.join('weight', 'weights_' + name + '_v2.hdf5')
# Training  mean_pred
earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(filepath=path1, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=2)
model.fit(trainX, trainY, epochs=150, batch_size=batch, shuffle=True, validation_split=0.1,
          callbacks=[earlyStopping, checkpoint])

# pdb.set_trace()
# model=load_model(path1)
# print(name)

# Evaluation
score = model.evaluate(testX, testY, batch_size=batch)
prediction = model.predict(testX, batch_size=batch, verbose=0)
mIoU = meanIoU(prediction, testY)
result = np.append(score, mIoU)
print('name, loss, accuracy, mIoU =  ', name, result)

# visualization
# path2=os.path.join('y_pred_'+name+'.npy')
# y_pred=prediction_reshape(prediction)
# np.save(path2,y_pred)
