import numpy as np
import logging
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, UpSampling2D
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename="particle_active_improved_8_1.log",
    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger = logging.getLogger("main")
num_actions = 2


class DiscriminativeEarlyStopping(Callback):

    def __init__(self, monitor='acc', threshold=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.improved = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current > self.threshold:
            if self.verbose > 0:
                logging.info("Epoch {e}: early stopping at accuracy {a}".format(
                    e=epoch, a=current))
            self.model.stop_training = True


class DelayedModelCheckpoint(Callback):

    def __init__(self, filepath, monitor='val_acc', delay=50, verbose=0, weights=False):

        super(DelayedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.delay = delay
        if self.monitor == 'val_acc':
            self.best = -np.Inf
        else:
            self.best = np.Inf
        self.weights = weights

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.monitor == 'val_acc':
            # import ipdb; ipdb.set_trace()
            print(logs)
            current = logs.get(self.monitor)
            print(current)
            if current >= self.best and epoch > self.delay:
                if self.verbose > 0:
                    logging.info('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                 ' saving model to %s'
                                 % (epoch, self.monitor, self.best,
                                    current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)
        else:
            current = logs.get(self.monitor)
            if current <= self.best and epoch > self.delay:
                if self.verbose > 0:
                    logging.info('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                 ' saving model to %s'
                                 % (epoch, self.monitor, self.best,
                                    current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(
            ser_model, gpus, cpu_relocation=False, cpu_merge=False)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def get_discriminative_model(input_shape):

    if np.sum(input_shape) < 30:
        width = 20
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(2, activation='softmax', name='softmax'))
    else:
        # print(input_shape)#5,5,5,128

        width = 256
        model = Sequential()

        model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same',
                         input_shape=input_shape, kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1)))
        model.add(Conv3D(64, (3, 3, 3), activation='relu',
                         padding='same', kernel_initializer='he_uniform'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1)))

        model.add(Flatten(input_shape=input_shape))
        # model.add(Dropout(0.5))
        model.add(Dense(width, activation='relu', use_bias=True,
                        kernel_initializer='TruncatedNormal'))
        # model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', use_bias=True,
                        kernel_initializer='TruncatedNormal'))
        # model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', use_bias=True,
                        kernel_initializer='TruncatedNormal'))
        # model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax', name='softmax',
                        kernel_initializer='TruncatedNormal'))

    return model


def get_model(input_shape, labels=2):

    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape,
                     padding='same', kernel_initializer='he_uniform'))
    #model.add(MaxPooling3D(pool_size=(2, 2,2)))
    model.add(Conv3D(32, (3, 3, 3), activation='relu',
                     padding='same', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', kernel_initializer='he_uniform'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', kernel_initializer='he_uniform'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', kernel_initializer='he_uniform'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           strides=(2, 2, 2), name='mark'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', name='embedding1',
                    kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.7))
    model.add(Dense(256, activation='relu', name='embedding2',
                    kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.7))
    model.add(Dense(labels, activation='softmax', name='softmax',
                    kernel_initializer='TruncatedNormal'))

    return model


def train_discriminative_model(labeled, unlabeled, input_shape, gpu=1):

    # create the binary dataset:
    y_L = np.zeros((labeled.shape[0], 1), dtype='int')
    y_U = np.ones((unlabeled.shape[0], 1), dtype='int')
    X_train = np.vstack((labeled, unlabeled))
    Y_train = np.vstack((y_L, y_U))
    Y_train = to_categorical(Y_train)
    print(Y_train)
    # build the model:
    model = get_discriminative_model(input_shape)

    # train the model:
    batch_size = 1024
    if np.max(input_shape) == 28:
        optimizer = optimizers.Adam(
            lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        epochs = 200
    elif np.max(input_shape) == 128:
        optimizer = optimizers.RMSprop(lr=0.0003, rho=0.9, epsilon=1e-06)
        batch_size = 32
        #optimizer = optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        epochs = 300  # TODO: was 200
    elif np.max(input_shape) == 512:
        optimizer = optimizers.Adam(
            lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # optimizer = optimizers.RMSprop()
        epochs = 500
    elif np.max(input_shape) == 32:
        optimizer = optimizers.Adam(
            lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        epochs = 500
    else:
        #optimizer = optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        optimizer = optimizers.RMSprop(lr=0.0003, rho=0.9, epsilon=1e-06)
        epochs = 300
        batch_size = 32

    # model.save('discriminator.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DiscriminativeEarlyStopping()]
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=callbacks,
              class_weight={0: float(X_train.shape[0]) / Y_train[Y_train == 0].shape[0],
                            1: float(X_train.shape[0]) / Y_train[Y_train == 1].shape[0]},
              verbose=2)
    # model.save('discriminator.h5')

    return model


def train_cryoet_model(
        args,
        X_train,
        Y_train,
        X_validation,
        Y_validation,
        checkpoint_path,
        gpu=1):

    if K.image_data_format() == 'channels_last':
        input_shape = (28, 28, 28, 1)
    else:
        input_shape = (1, 40, 40, 40)

    model = get_model(input_shape=input_shape, labels=args.classes)
    optimizer = optimizers.Adam(
        lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DelayedModelCheckpoint(
        filepath=checkpoint_path, verbose=1, weights=True)]

    if gpu > 1:
        gpu_model = ModelMGPU(model, gpus=gpu)
        gpu_model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer, metrics=['accuracy'])
        gpu_model.fit(X_train, Y_train,
                      epochs=300,
                      batch_size=64,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        del model
        del gpu_model

        model = get_model(input_shape=input_shape, labels=2)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
        model.load_weights(checkpoint_path)
        return model

    else:
        model.fit(X_train, Y_train,
                  epochs=200,
                  batch_size=128,
                  shuffle=True,
                  validation_data=(X_validation, Y_validation),
                  callbacks=callbacks,
                  verbose=2)
        # model.save_weights(checkpoint_path)
        model.load_weights(checkpoint_path)
        return model
