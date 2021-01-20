import keras
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D,\
    Cropping3D, Dropout, Add, Reshape, concatenate
from keras.models import Sequential, Model, load_model
from keras.layers.core import Activation
import numpy as np


# Model  2AWB
def FCN8(img_shape, class_n=None):
    input_shape = (None, img_shape[0], img_shape[1], img_shape[2], 1)

    # use relu activation for hidden layer to guarantee non-negative
    # outputs are passed to the max pooling layer. In such case,
    # as long as the output layer is linear activation,
    # the network can still accomodate negative image intendities,
    # just matter of shift back using the bias term
    input_img = Input(shape=input_shape[1:])

    conv1_1 = Conv3D(32, 3, padding='same', activation='relu')(input_img)
    conv1_2 = Conv3D(32, 3, padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv1_2)

    conv2_1 = Conv3D(64, 3, padding='same', activation='relu')(pool1)
    conv2_2 = Conv3D(64, 3, padding='same', activation='relu')(conv2_1)
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv2_2)

    conv3_1 = Conv3D(128, 3, padding='same', activation='relu')(pool2)
    conv3_2 = Conv3D(128, 3, padding='same', activation='relu')(conv3_1)
    # if use_dilated:
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv3_2)

    # fully conv
    fc4 = Conv3D(1024, 1, padding='same', activation='relu')(pool3)
    drop5 = Dropout(0.7)(fc4)

    fc6 = Conv3D(1024, 1, padding='same', activation='relu')(drop5)
    drop7 = Dropout(0.7)(fc6)

    # upsampling
    score1 = Conv3D(2, 1, padding='same')(drop7)
    upscore1 = UpSampling3D((2, 2, 2))(score1)
    score_pool1 = Conv3D(2, 1, padding='same')(pool2)
    # upscore1,score_pool1 =crop( upscore1 , score_pool1 , input_img)

    score2 = Add()([upscore1, score_pool1])
    upscore2 = UpSampling3D((2, 2, 2))(score2)
    score_pool2 = Conv3D(2, 1, padding='same')(pool1)
    # upscore2,score_pool2 =crop( upscore2 , score_pool2 , input_img)

    score3 = Add()([upscore2, score_pool2])
    upscore3 = UpSampling3D((2, 2, 2))(score3)
    # upscore4=checker(upscore3,input_img)

    output1 = Reshape((np.prod(img_shape), 2))(upscore3)
    output = Activation('softmax')(output1)

    model = Model(inputs=input_img, outputs=output)

    return model


# Model  2AWB
def FCN1(img_shape, class_n=None):
    input_shape = (None, img_shape[0], img_shape[1], img_shape[2], 1)

    # use relu activation for hidden layer to guarantee non-negative outputs
    # are passed to the max pooling layer. In such case,
    # as long as the output layer is linear activation, the network can still
    # accomodate negative image intendities, just matter of shift back using the bias term
    input_img = Input(shape=input_shape[1:])

    conv1_1 = Conv3D(32, 3, padding='same', activation='relu')(input_img)
    conv1_2 = Conv3D(32, 3, padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv1_2)

    conv2_1 = Conv3D(64, 3, padding='same', activation='relu')(pool1)
    conv2_2 = Conv3D(64, 3, padding='same', activation='relu')(conv2_1)
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv2_2)

    conv3_1 = Conv3D(128, 3, padding='same', activation='relu')(pool2)
    conv3_2 = Conv3D(128, 3, padding='same', activation='relu')(conv3_1)
    # if use_dilated:
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv3_2)

    # fully conv
    fc4 = Conv3D(1024, 1, padding='same', activation='relu')(pool3)
    drop5 = Dropout(0.7)(fc4)

    fc6 = Conv3D(1024, 1, padding='same', activation='relu')(drop5)
    drop7 = Dropout(0.7)(fc6)

    # upsampling
    score1 = Conv3D(2, 1, padding='same')(drop7)
    upscore1 = UpSampling3D((2, 2, 2))(score1)

    # upscore1,score_pool1 =crop( upscore1 , score_pool1 , input_img)

    score2 = Conv3D(2, 1, padding='same')(upscore1)
    score3 = Conv3D(2, 1, padding='same')(score2)
    upscore2 = UpSampling3D((2, 2, 2))(score3)

    # upscore2,score_pool2 =crop( upscore2 , score_pool2 , input_img)

    score4 = Conv3D(2, 1, padding='same')(upscore2)
    score5 = Conv3D(2, 1, padding='same')(score4)
    upscore3 = UpSampling3D((2, 2, 2))(score4)
    # upscore4=checker(upscore3,input_img)

    output1 = Reshape((np.prod(img_shape), 2))(upscore3)
    output = Activation('softmax')(output1)

    model = Model(inputs=input_img, outputs=output)

    return model


def FCN_ed(img_shape, class_n=None):
    input_shape = (None, img_shape[0], img_shape[1], img_shape[2], 1)
    # use relu activation for hidden layer to guarantee non-negative
    # outputs are passed to the max pooling layer. In such case,
    # as long as the output layer is linear activation, the network can
    # still accomodate negative image intendities,
    # just matter of shift back using the bias term
    input_img = Input(shape=input_shape[1:])

    conv1_1 = Conv3D(32, 3, padding='same', activation='relu')(input_img)
    conv1_2 = Conv3D(32, 3, padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv1_2)

    conv2_1 = Conv3D(64, 3, padding='same', activation='relu')(pool1)
    conv2_2 = Conv3D(64, 3, padding='same', activation='relu')(conv2_1)
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv2_2)

    conv3_1 = Conv3D(128, 3, padding='same', activation='relu')(pool2)
    conv3_2 = Conv3D(128, 3, padding='same', activation='relu')(conv3_1)
    # if use_dilated:
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv3_2)

    # # fully conv
    # fc4 = Conv3D(1024, 1, padding='same', activation='relu')(pool3)
    # drop5 = Dropout(0.7)(fc4)

    # fc6 = Conv3D(1024, 1, padding='same', activation='relu')(drop5)
    # drop7 = Dropout(0.7)(fc6)

    # upsampling
    score1_1 = Conv3D(128, 1, padding='same', activation='relu')(pool3)
    upscore1 = UpSampling3D((2, 2, 2))(score1_1)

    score_pool1 = Conv3D(128, 1, padding='same')(pool2)
    score1 = Add()([upscore1, score_pool1])

    score2_1 = Conv3D(64, 1, padding='same', activation='relu')(score1)
    upscore2 = UpSampling3D((2, 2, 2))(score2_1)

    score_pool2 = Conv3D(64, 1, padding='same')(pool1)
    score2 = Add()([upscore2, score_pool2])

    score3_1 = Conv3D(32, 1, padding='same', activation='relu')(score2)
    upscore3 = UpSampling3D((2, 2, 2))(score3_1)
    # upscore4=checker(upscore3,input_img)
    upscore4 = Conv3D(2, 1, padding='same')(upscore3)
    output1 = Reshape((np.prod(img_shape), 2))(upscore4)
    output = Activation('softmax')(output1)

    model = Model(inputs=input_img, outputs=output)

    return model


# Model  2AWB
def FCN_aspp(img_shape, class_n=None):
    input_shape = (None, img_shape[0], img_shape[1], img_shape[2], 1)

    # use relu activation for hidden layer to guarantee non-negative outputs
    # are passed to the max pooling layer. In such case, as long as
    # the output layer is linear activation, the network can still accomodate
    # negative image intendities, just matter of shift back using the bias term
    input_img = Input(shape=input_shape[1:])

    conv1_1 = Conv3D(32, 3, padding='same', activation='relu')(input_img)
    conv1_2 = Conv3D(32, 3, padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv1_2)

    conv2_1 = Conv3D(64, 3, padding='same', activation='relu')(pool1)
    conv2_2 = Conv3D(64, 3, padding='same', activation='relu')(conv2_1)
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv2_2)

    conv3_1 = Conv3D(128, 3, padding='same', activation='relu', dilation_rate=2)(pool2)

    fcp1 = Conv3D(128, 1, padding='same', activation='relu')(conv3_1)
    fcp2 = Conv3D(128, 3, padding='same', activation='relu', dilation_rate=2)(conv3_1)
    fcp3 = Conv3D(128, 3, padding='same', activation='relu', dilation_rate=4)(conv3_1)
    cancate = concatenate([fcp1, fcp2, fcp3, conv3_1])

    # upsampling
    upscore1 = UpSampling3D((4, 4, 4))(cancate)

    # score_pool1=Conv3D(512, 1, padding='same')(pool1)
    # score1=Add()([upscore1,score_pool1])

    # upscore2=UpSampling3D((2,2,2))(score1)

    # upscore4=checker(upscore3,input_img)
    upscore4 = Conv3D(2, 1, padding='same')(upscore1)
    output1 = Reshape((np.prod(img_shape), 2))(upscore4)
    output = Activation('softmax')(output1)

    model = Model(inputs=input_img, outputs=output)

    return model


# test the dilated conv
def FCN_ed2(img_shape, class_n=None):
    input_shape = (None, img_shape[0], img_shape[1], img_shape[2], 1)

    # use relu activation for hidden layer to guarantee non-negative outputs
    # are passed to the max pooling layer. In such case, as long as the output
    # layer is linear activation, the network can still accomodate negative
    # image intendities, just matter of shift back using the bias term
    input_img = Input(shape=input_shape[1:])

    conv1_1 = Conv3D(32, 3, padding='same', activation='relu')(input_img)
    conv1_2 = Conv3D(32, 3, padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv1_2)

    conv2_1 = Conv3D(64, 3, padding='same', activation='relu')(pool1)
    conv2_2 = Conv3D(64, 3, padding='same', activation='relu')(conv2_1)
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv2_2)

    conv3_1 = Conv3D(128, 3, padding='same', activation='relu')(pool2)
    conv3_2 = Conv3D(128, 3, padding='same', activation='relu')(conv3_1)
    # if use_dilated:
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(conv3_2)

    # upsampling
    score1_1 = Conv3D(128, 1, padding='same', activation='relu', dilation_rate=2)(pool3)
    upscore1 = UpSampling3D((2, 2, 2))(score1_1)

    score_pool1 = Conv3D(128, 1, padding='same')(pool2)
    score1 = Add()([upscore1, score_pool1])

    score2_1 = Conv3D(64, 1, padding='same', activation='relu')(score1)
    upscore2 = UpSampling3D((2, 2, 2))(score2_1)

    score_pool2 = Conv3D(64, 1, padding='same')(pool1)
    score2 = Add()([upscore2, score_pool2])

    score3_1 = Conv3D(32, 1, padding='same', activation='relu')(score2)
    upscore3 = UpSampling3D((2, 2, 2))(score3_1)
    # upscore4=checker(upscore3,input_img)
    upscore4 = Conv3D(2, 1, padding='same')(upscore3)
    output1 = Reshape((np.prod(img_shape), 2))(upscore4)
    output = Activation('softmax')(output1)

    model = Model(inputs=input_img, outputs=output)

    return model
