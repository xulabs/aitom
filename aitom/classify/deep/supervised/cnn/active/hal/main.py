import pickle
import os
import logging
import argparse
import tensorflow as tf
from utils import *
from models import *
from query_methods import *
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


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('--lambda_e', type=float, default=1)
    p.add_argument('--classes', type=float, default=10)
    p.add_argument('--experiment_index', type=int, default=0)
    p.add_argument('--data_type', type=str, default='cryoet')
    p.add_argument('--batch_size', type=int, default=400)
    p.add_argument('--initial_size', type=int, default=400)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--method', type=str,
                   default='HAL')
    p.add_argument('--experiment_folder', type=str, default='EXP',
                   help="folder where the experiment results will be saved")
    p.add_argument('--gpu', '-gpu', type=int, default=0)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args


def load_data(path='./data_particle.npz'):
    # f = np.load(path)
    # x_train, y_train = f['x_train'], f['y_train']
    # x_test, y_test = f['x_test'], f['y_test']
    # f.close()
    x_train = np.random.randn(5000, 28, 28, 28)
    y_train = np.random.randint(10, size=50000)
    x_test = np.random.randn(10000, 28, 28, 28)
    y_test = np.random.randint(10, size=10000)
    return (x_train, y_train), (x_test, y_test)


def load_cryoet():
    (x_train, y_train), (x_test, y_test) = load_data()

    if K.image_data_format() == 'channels_last':
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 28, 1))
    else:
        x_train = x_train.reshape((x_train.shape[0], 3, 128, 128))
        x_test = x_test.reshape((x_test.shape[0], 3, 128, 128))

    # shuffle the data:
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]

    return (x_train, y_train), (x_test, y_test)


def evaluate_sample(
        args,
        training_function,
        X_train,
        Y_train,
        X_test,
        Y_test,
        checkpoint_path):

    # shuffle the training set:
    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # create the validation set:
    X_validation = X_train[:int(0.2 * X_train.shape[0])]
    Y_validation = Y_train[:int(0.2 * Y_train.shape[0])]
    X_train = X_train[int(0.2 * X_train.shape[0]):]
    Y_train = Y_train[int(0.2 * Y_train.shape[0]):]

    # train and evaluate the model:
    model = training_function(
        args,
        X_train, Y_train,
        X_validation, Y_validation,
        checkpoint_path, gpu=args.gpu)
    if args.data_type in ['imdb', 'wiki']:
        acc = model.evaluate(X_test, Y_test, verbose=0)
    else:
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    return acc, model


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    # print(y)
    input_shape = y.shape
    # print(input_shape)
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    # print(y)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


if __name__ == '__main__':

    # parse the arguments:
    args = parse_input()

    # load the dataset:
    if args.data_type == 'cryoet':
        (X_train, Y_train), (X_test, Y_test) = load_cryoet()
        num_labels = 2
        if K.image_data_format() == 'channels_last':
            input_shape = (28, 28, 28, 1)
        else:
            input_shape = (1, 40, 40, 40)
        evaluation_function = train_cryoet_model

    # make categorical:
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # set the first query method:
    if args.method == 'HAL':
        method = HAL

    checkpoint_path = create_save_folder(args, method)

    labeled_idx = np.asarray(list(range(0, args.initial_size)))
    logging.info(str(args.initial_size) + ' samples are initialized for training.....')

    # run the experiment:
    accuracies = []
    entropies = []
    label_distributions = []
    queries = []
    acc, model = evaluate_sample(
        args,
        evaluation_function,
        X_train[labeled_idx, :],
        Y_train[labeled_idx],
        X_test,
        Y_test,
        checkpoint_path)

    query_method.update_model(model)
    accuracies.append(acc)
    logging.info("Test Accuracy Is " + str(acc))
    for i in range(args.iterations):

        # get the new indices from the algorithm
        old_labeled = np.copy(labeled_idx)
        labeled_idx = query_method.query(
            X_train, Y_train, labeled_idx, args.batch_size)

        # calculate and store the label entropy:
        new_idx = labeled_idx[np.logical_not(
            np.isin(labeled_idx, old_labeled))]
        new_labels = Y_train[new_idx]
        new_labels /= np.sum(new_labels)
        new_labels = np.sum(new_labels, axis=0)
        entropy = -np.sum(new_labels * np.log(new_labels + 1e-10))
        entropies.append(entropy)
        label_distributions.append(new_labels)
        queries.append(new_idx)

        # evaluate the new sample:
        acc, model = evaluate_sample(
            args,
            evaluation_function,
            X_train[labeled_idx],
            Y_train[labeled_idx],
            X_test,
            Y_test,
            checkpoint_path)
        query_method.update_model(model)
        accuracies.append(acc)
        logging.info("Test Accuracy Is " + str(acc))

        # save the sampled indexes every iteration for retraining.

        with open(results_path_each_iteration, 'wb') as f:
            pickle.dump([i, labeled_idx], f)
            logging.info("Saved current labeled indexes to " +
                         results_path_each_iteration)

    # save the results:
    with open(results_path, 'wb') as f:
        pickle.dump([accuracies, args.initial_size, args.batch_size], f)
        logging.info("Saved results to " + results_path)
    with open(entropy_path, 'wb') as f:
        pickle.dump([entropies, label_distributions, queries], f)
        logging.info("Saved entropy statistics to " + entropy_path)
