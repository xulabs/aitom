import os, sys
import numpy as np
import aitom.io.file as AIF

from os.path import join as op_join
from sklearn.metrics import mean_squared_error
from .auto_classifier_model import auto_classifier_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam


# preprocess data into training set (labeled data + unlabeled data),
# validation set and testing set
def preprocess(d, num_of_class):
    classes = {'1KP8_data': 0, '5T2C_data': 1}

    # percentage of training data
    percent_of_training = 0.8
    # percentage of labeled data in training set
    percent_of_labels = 0.5
    # percentage of validation data in training set
    percent_of_validation = 0.1

    labels = []
    x_train = []
    masks = []
    masks_auto = []
    x_test = []
    test_labels = []
    data_validation = []
    labels_validation = []

    for key in classes:
        nums = 0
        img_list = d[key]
        np.random.seed(0)
        np.random.shuffle(img_list)
        for img in img_list:
            nums += 1
            label = np.zeros(num_of_class)
            # used to control the number of the labeled samples.
            mask = np.zeros(num_of_class)
            if 1.0 * nums / len(d[key]) <= percent_of_labels * percent_of_training:
                # training samples with labels
                label[classes[key]] = 1
                x_train.append(np.expand_dims(img, -1))
                labels.append(label)
                masks.append(True)
                masks_auto.append(True)
            elif percent_of_training * (1 - percent_of_validation) >= (1.0 * nums / len(d[key]))\
                    > percent_of_labels * percent_of_training:
                # training samples without labels
                label[classes[key]] = 1
                x_train.append(np.expand_dims(img, -1))
                labels.append(label)
                masks.append(False)
                masks_auto.append(True)
            elif percent_of_training * (1 - percent_of_validation) < (1.0 * nums / len(d[key]))\
                    <= percent_of_training:
                # validation set
                label[classes[key]] = 1
                data_validation.append(np.expand_dims(img, -1))
                labels_validation.append(label)
            else:
                # testing set
                label[classes[key]] = 1
                x_test.append(np.expand_dims(img, -1))
                test_labels.append(label)

    x_train = np.array(x_train)
    labels = np.array(labels)
    # the masks for training and validation.
    masks = np.array(masks, dtype=np.bool)
    x_test = np.array(x_test)
    test_labels = np.array(test_labels)
    masks_auto = np.array(masks_auto, dtype=np.bool)
    data_validation = np.array(data_validation)
    labels_validation = np.array(labels_validation)
    return [x_train, x_test, data_validation], [labels, test_labels, labels_validation], [masks, masks_auto]


def run_auto_classifier(d, option, out_dir):
    num_of_class = 2
    all_data, all_labels, all_masks = preprocess(d, num_of_class)
    # training, testing and validation data
    x_train, x_test, data_validation = all_data
    # training, testing and validation labels
    labels, test_labels, labels_validation = all_labels
    # masks
    masks, masks_auto = all_masks

    model_dir = op_join(out_dir, 'model')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_autoclassifier_checkpoint_file = op_join(model_dir, 'model-autoclassifier--weights--best.h5')

    if option == 'train':
        model = auto_classifier_model(img_shape=x_train[0].shape, num_of_class=num_of_class)
        # choose a proper lr to control convergance speed, and val_loss
        adam = Adam(lr=0.0003, beta_1=0.9, decay=0.001/500)
        masks_auto_training = masks_auto
        # sequential_1: autoencoder output. dense_4: classifier output
        losses = {'sequential_1': "mean_squared_error",
                  'dense_4': "categorical_crossentropy"}
        lossWeights = {'sequential_1': 1.0, 'dense_4': 1.0}
        model.compile(optimizer=adam, loss=losses, loss_weights=lossWeights,
                      metrics={'sequential_1': "mean_squared_error",
                               'dense_4': "accuracy"})

        if os.path.isfile(model_autoclassifier_checkpoint_file):
            print('loading previous best weights', model_autoclassifier_checkpoint_file)
            model.load_weights(model_autoclassifier_checkpoint_file)

        earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_autoclassifier_checkpoint_file,
                                     monitor='val_dense_4_acc', verbose=1,
                                     save_best_only=True, mode='auto')
        model.fit(x_train, [x_train, labels], epochs=1, batch_size=16, shuffle=True,
                  sample_weight={'dense_4': masks, 'sequential_1': masks_auto_training},
                  validation_data=(data_validation, [data_validation, labels_validation]),
                  callbacks=[checkpoint, earlyStopping])
    else:
        model = auto_classifier_model(img_shape=x_train[0].shape, num_of_class=num_of_class)
        model.load_weights(model_autoclassifier_checkpoint_file)
        x_rec, classification_testing = model.predict([x_test])
        test_prediction = np.argmax(classification_testing, axis=1)
        test_real_class = np.argmax(test_labels, axis=1)
        true = 0.
        all_sample = float( len(test_prediction))
        for i in range(len(test_prediction)):
            if test_prediction[i] == test_real_class[i]:
                true += 1.
        testing_accuracy = true / all_sample
        print("Classification Accuracy: %f" % testing_accuracy)
        mse_error = mean_squared_error(x_rec.flatten(), x_test.flatten())
        print("Reconstruction Error: %f" % mse_error)


if __name__ == "__main__":
    d = AIF.pickle_load(sys.argv[1])
    option = sys.argv[2]
    run_auto_classifier(d=d, option=option, out_dir=os.getcwd())
