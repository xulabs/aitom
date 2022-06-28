
from tensorflow import keras
# import keras
from keras.models import Sequential, Model, load_model
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
# import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, UpSampling2D, Convolution3D, ZeroPadding3D
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, BatchNormalization, AveragePooling3D, Concatenate
import os
import sys
import argparse
import tensorflow as tf
import pickle
from keras.backend.tensorflow_backend import set_session
import sklearn
from prdc import compute_prdc
from dataLoader import *
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score, mean_squared_error

validation_split = 0.8 #0.8 train ,0.2 val
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.99
#set_session(tf.Session(config=config))


def read_data(pic_sub_dir):
    with open(pic_sub_dir, "rb") as pic:
        pic_sub = pickle.load(pic, encoding='latin1')

    ribosome = np.array(pic_sub['ribosome']).reshape((-1, 40, 40, 40, 1))
    membrane = np.array(pic_sub['membrane']).reshape((-1, 40, 40, 40, 1))
    TRiC = np.array(pic_sub['TRiC']).reshape((-1, 40, 40, 40, 1))
    proteasome_s = np.array(
        pic_sub['proteasome_s']).reshape((-1, 40, 40, 40, 1))
    return ribosome, membrane, TRiC, proteasome_s

def split_dataset(data, label, validation_split):
    num = data.shape[0]
    train_num = int(num*validation_split)
    test_num = num-train_num
    trainlabel = np.ones(train_num)*label
    testlabel = np.ones(test_num)*label
    np.random.shuffle(data)
    train_data = data[0:train_num, :]
    test_data = data[train_num:, :]
    return train_data, trainlabel, test_data, testlabel

def preprocess(X_all):         
    for i in range(X_all.shape[0]):
        img = X_all[i]
        img = (img-img.min())/(img.max()-img.min())
        X_all[i]=img
    #X_all = (X_all-MIN)/((MAX-MIN)/2)-1
    return X_all


def c3d(image_size, num_labels=4):
    num_channels = 1
    model = Sequential()
    input_shape = (image_size, image_size, image_size,
                   num_channels)  # l, h, w, c
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1a',
                            input_shape=input_shape))
    #model.add(Convolution3D(64, 3, 3, 3, activation='relu',
    #                        border_mode='same', name='conv1b'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2a'))
    #model.add(Convolution3D(128, 3, 3, 3, activation='relu',
    #                        border_mode='same', name='conv2b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a'))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b'))

    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))

    model.add(Flatten())
    #model.add(BatchNormalization())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(num_labels, activation='softmax', name='fc8'))

    return model
def train():
    input_shape = (40,40,40,1)

    x_train, y_train, x_val, y_val = get_picdata(data_path='../data/new_4_classes.pickle')

    print(y_val[0:30])
    y_train = np_utils.to_categorical(y_train, 4)
    y_val = np_utils.to_categorical(y_val, 4)

    #  test



    #Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))

    #Create labels as one-hot vectors
    #labels_train = keras.utils.np_utils.to_categorical(labels_train, num_classes=10)
    #labels_test = keras.utils.np_utils.to_categorical(labels_test, num_classes=10)


    #Create the model

    model = c3d(40,4)
    #model = fc_model(input_shape,4)

    #Model parameters
    batch_size = 32
    import keras.optimizers as KOP
    #kop = KOP.SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)
    #Compile model using croo entropy as loss and adam as optimizer
    optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #Train model using input of clean and corrupted data and fit to clean reconstructions only
    # Train data and label :   x_train, y_train  
    # Val data and label :   x_val, y_val
    # Test( Real sub ) data and label :   data_test, labels_test
    model.fit(x_train, y_train, validation_data=(x_val, y_val), #(data_test, labels_test),
            epochs=1, verbose=1, batch_size=batch_size, shuffle=True)


    #Save the model
    model.save('../data/newsub-4.h5')  # simu_c3d.h5


def test():
    model = load_model('../data/trainonsub.h5')
    ribosome, membrane, TRiC, proteasome_s = read_data("../data/new_4_classes.pickle")
    id2label = {"ribosome": 0, "membrane": 1, "TRiC": 2, "proteasome_s": 3}
    x_test = np.concatenate((ribosome,membrane,TRiC,proteasome_s),axis=0)
    x_test = preprocessz(x_test)
    print(x_test.shape)
    y_test = np.concatenate((np.ones(
        ribosome.shape[0])*id2label["ribosome"], np.ones(membrane.shape[0])*id2label["membrane"], np.ones(TRiC.shape[0])*id2label["TRiC"], np.ones(proteasome_s.shape[0])*id2label["proteasome_s"]))
    y_tests = np_utils.to_categorical(y_test, 4)
    print(y_test.shape)

    pred_prob = model.predict(x_test)

    pred_labels = pred_prob.argmax(axis=-1)

    #Evaluate
    scores = model.evaluate(x_test, y_tests)  # (x_val, y_val)
    com_scores = completeness_score(y_test, pred_labels)
    hom_scores = homogeneity_score(y_test, pred_labels)
    v_scores = v_measure_score(y_test, pred_labels)


    #Print accuracy
    
    print ("Accuracy in simulator subtomograms: {} %".format(scores[1]*100))
    print("com_scores: {}".format(com_scores))
    print("hom_scores:  {}".format(hom_scores))
    print("v_scores:  {}".format(v_scores))

def evalaute():
    model = load_model('../data/trainonsub.h5')
    model_name = "wgan4_sim_lr2e-5"
    y_test = np.load("../result/{}/best_acc_fake_subtomogram_test.npy".format(model_name))#1KP8
    x = np.load("../result/{}/density_map_test.npy".format(model_name))#1KP8
    y = np.load("../result/{}/subtomogram_test.npy".format(model_name))#subtomogram
    x_label = np.load("../result/{}/density_map_label_test.npy".format(model_name))
    labels_test = np_utils.to_categorical(x_label, 4)

    # pic_sub = read_data("../data/new_4_classes.pickle")
    # label2id = {"ribosome": 0, "membrane": 1, "TRiC": 2, "proteasome_s": 3}

    pred_prob = model.predict(y_test)

    pred_labels = pred_prob.argmax(axis=-1)

    #Evaluate
    scores = model.evaluate(y_test, labels_test)  # (x_val, y_val)
    # com_scores = completeness_score(y_test, pred_labels)
    # hom_scores = homogeneity_score(y_test, pred_labels)
    # v_scores = v_measure_score(y_test, pred_labels)

    valid_acc = model.evaluate(y, labels_test)
    #Print accuracy
    print ("Accuracy for real sub: {} %".format(valid_acc[1]*100))
    print ("Accuracy in generated subtomograms: {} %".format(scores[1]*100))

    metrics = compute_prdc(real_features=y.reshape((x.shape[0],-1)),
                        fake_features=y_test.reshape((x.shape[0],-1)),
                        nearest_k=5)
              
    # compute psnr
    mse = mean_squared_error(y.reshape((y_test.shape[0],-1)), y_test.reshape((y_test.shape[0],-1)))
    psnr = 10. * np.log10(1. / mse)

    print("PSNR is {}".format(psnr))
    print(metrics)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # train()
    # test()
    evalaute()





