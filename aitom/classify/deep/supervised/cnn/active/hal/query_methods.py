import gc
import logging
from keras.models import Model
from models import *
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


def get_unlabeled_idx(X_train, labeled_idx):
    return np.arange(X_train.shape[0])[np.logical_not(
        np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


class QueryMethod:

    def __init__(self, model,
                 args,
                 input_shape=(28, 28, 28),
                 num_labels=2,
                 gpu=1):
        self.model = model
        self.input_shape = input_shape
        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, X_train, Y_train, labeled_idx, amount):
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model


class HAL(QueryMethod):

    def __init__(self, model, args, input_shape, num_labels, gpu):
        super().__init__(model, args, input_shape, num_labels, gpu)
        self.args = args
        self.sub_batches = self.args.subset_number

    def query(self, X_train, Y_train, labeled_idx, amount):

        # subsample from the unlabeled set:
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled_idx = np.random.choice(unlabeled_idx,
                                         np.min([labeled_idx.shape[0] * 100,
                                                 unlabeled_idx.size]),
                                         replace=False)
        # print(self.model.input.shape)
        embedding_model = Model(inputs=self.model.input,
                                outputs=self.model.get_layer('mark').output)
        representation = embedding_model.predict(
            X_train)  # .reshape((X_train.shape[0], -1, 1))
        logging.info(representation.shape)

        # iteratively sub-sample using the discriminative sampling routine:
        labeled_so_far = 0
        sub_sample_size = int(amount / self.sub_batches)
        while labeled_so_far < amount:
            if labeled_so_far + sub_sample_size > amount:
                sub_sample_size = amount - labeled_so_far

            model = train_discriminative_model(
                representation[labeled_idx],
                representation[unlabeled_idx],
                representation[0].shape, gpu=self.gpu)
            predictions = model.predict(representation[unlabeled_idx])

            # entropy metric.
            predictions_entropy = self.model.predict(X_train[unlabeled_idx])
            unlabeled_predictions_entropy = - np.sum(
                predictions_entropy * np.log(
                    predictions_entropy + 1e-10), axis=1)

            predictions_temp = predictions[:, 1] \
                + self.args.lambda_e * unlabeled_predictions_entropy

            # end.
            selected_indices = np.argpartition(
                predictions_temp, -sub_sample_size)[-sub_sample_size:]
            labeled_idx = np.hstack(
                (labeled_idx, unlabeled_idx[selected_indices]))
            labeled_so_far += sub_sample_size
            unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
            unlabeled_idx = np.random.choice(unlabeled_idx, np.min(
                [labeled_idx.shape[0] * 100, unlabeled_idx.size]), replace=False)

            # delete the model to free GPU memory:
            del model
            gc.collect()
        del embedding_model
        gc.collect()

        return labeled_idx
