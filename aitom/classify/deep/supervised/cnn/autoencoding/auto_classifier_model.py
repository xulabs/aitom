import numpy as np
import keras.backend as K
from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, UpSampling3D,\
    Reshape, Flatten, Activation, Multiply, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras import regularizers


def conv_block(x, nb_filter, nb0, nb1, nb2, border_mode='same',
               subsample=(1, 1, 1), bias=True, batch_norm=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Convolution3D(nb_filter, nb0, nb1, nb2, subsample=subsample,
                      border_mode=border_mode, bias=bias)(x)
    if batch_norm:
        assert not bias
        x = BatchNormalization(axis=channel_axis)(x)
    else:
        assert bias

    x = Activation('relu')(x)

    return x


def auto_classifier_model(img_shape, encoding_dim=128, NUM_CHANNELS=1, num_of_class=2):
    input_shape = (None, img_shape[0], img_shape[1], img_shape[2], NUM_CHANNELS)
    mask_shape = (None, num_of_class)

    # use relu activation for hidden layer to guarantee non-negative outputs
    # are passed to the max pooling layer. In such case, as long as the output layer
    # is linear activation, the network can still accomodate negative image intendities,
    # just matter of shift back using the bias term
    input_img = Input(shape=input_shape[1:])
    mask = Input(shape=mask_shape[1:])
    x = input_img

    x = conv_block(x, 32, 3, 3, 3)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    x = conv_block(x, 32, 3, 3, 3)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    # x.get_shape() returns a list of tensorflow.python.framework.tensor_shape.Dimension objects
    encoder_conv_shape = [_.value for _ in x.get_shape()]
    x = Flatten()(x)
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)
    encoder = Model(inputs=input_img, outputs=encoded)

    x = BatchNormalization()(x)
    x = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_of_class, activation='softmax')(x)

    prob = x
    # classifier output
    classifier = Model(inputs=input_img, outputs=prob)

    input_img_decoder = Input(shape=encoder.output_shape[1:])
    x = input_img_decoder
    x = Dense(np.prod(encoder_conv_shape[1:]), activation='relu')(x)
    x = Reshape(encoder_conv_shape[1:])(x)

    x = UpSampling3D((2, 2, 2))(x)
    x = conv_block(x, 32, 3, 3, 3)

    x = UpSampling3D((2, 2, 2))(x)
    x = conv_block(x, 32, 3, 3, 3)
    x = Convolution3D(1, (3, 3, 3), activation='linear', padding='same')(x)

    decoded = x
    # autoencoder output
    decoder = Model(inputs=input_img_decoder, outputs=decoded)

    autoencoder = Sequential()
    for l in encoder.layers:
        autoencoder.add(l)
    last = None
    for l in decoder.layers:
        last = l
        autoencoder.add(l)

    decoded = autoencoder(input_img)

    auto_classifier = Model(inputs=input_img, outputs=[decoded, prob])
    auto_classifier.summary()
    return auto_classifier
