import h5py
import os
import numpy as np
from functools import partial

from .master_models import *

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
# from keras.layers.convolutional import Convolution2D, Conv2DTranspose
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
# from keras.datasets import mnist
from keras import backend as K


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def train(X_train, paths):
    assert ('generator_save_path' in paths and 'discriminator_save_path' in paths)
    if not os.path.isdir(paths['generator_save_path']):
        os.makedirs(paths['generator_save_path'])
    if not os.path.isdir(paths['discriminator_save_path']):
        os.makedirs(paths['discriminator_save_path'])

    # Now we initialize the generator and discriminator.
    print("Initializing networks... ")
    generator = make_generator()
    discriminator = make_discriminator()

    # The generator_model is used when we want to train the generator layers.
    # As such, we ensure that the discriminator layers are not trainable.
    # Note that once we compile this model, updating .trainable will have no effect within it. As such, it
    # won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
    # as we compile the generator_model first.
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generator_input = Input(shape=(GEN_INPUT_DIM,))
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(LR, beta_1=BETA_1, beta_2=BETA_2),
                            loss=[wasserstein_loss])

    # Now that the generator_model is compiled, we can make the discriminator layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
    # The noise seed is run through the generator model to get generated images. Both real and generated images
    # are then run through the discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples = Input(shape=X_train.shape[1:])
    generator_input_for_discriminator = Input(shape=(GEN_INPUT_DIM,))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
    # We then run these samples through the discriminator as well. Note that we never really use the discriminator
    # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input averaged samples to get gradients. However,
    # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
    # of the function with the averaged samples here.
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

    # Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
    # real samples and generated samples before passing them to the discriminator: If we had, it would create an
    # output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
    # would have only BATCH_SIZE samples.

    # If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
    # samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
    discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
    # samples, and the gradient penalty loss for the averaged samples.
    discriminator_model.compile(optimizer=Adam(LR, beta_1=BETA_1, beta_2=BETA_2),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
    # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
    # gradient_penalty loss function and is not used.
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    transform_vector = get_random_vector(10)
    discriminator_loss = []
    generator_loss = []
    for epoch in range(1, NUM_EPOCHS + 1):
        print("Epoch: ", epoch)
        print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
            print("Batch %d of %d" % (i, X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO)))
            np.random.shuffle(X_train)
            discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]

            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                noise = get_random_vector(BATCH_SIZE)
                discriminator_batch_loss = discriminator_model.train_on_batch([image_batch, noise],
                                                                              [positive_y, negative_y,
                                                                               dummy_y])

            generator_batch_loss = generator_model.train_on_batch(
                get_random_vector(BATCH_SIZE),
                positive_y)
            print('generator loss', generator_batch_loss, 'discriminator loss', discriminator_batch_loss)
            generator_loss.append(generator_batch_loss)
            discriminator_loss.append(discriminator_batch_loss)

        if epoch % 20 == 0:
            suffix = 'epoch_' + format(epoch, '#02d')
            generator.save_weights(os.path.join(paths['generator_save_path'], suffix))
            discriminator.save_weights(os.path.join(paths['discriminator_save_path'], suffix))


# Load the image data, reshape it and normalize it to the range [-1, 1]
def load_my_data(data_source):
    with h5py.File(data_source, 'r') as data_file:
        X_train = data_file['interpolated_shapes'][:]
        print("Loaded data of size %s from %s" % (X_train.shape, data_source))
        for i in range(len(X_train)):
            X_train[i] = (X_train[i] - X_train[i].min()) / (X_train[i].max() - X_train[i].min())
        np.random.shuffle(X_train)
        X_train = np.reshape(X_train, (X_train.shape + (1,)))
        return X_train[:, :, :, :, :]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="number of training epochs", type=int)
    parser.add_argument("--data", help="path to training data", type=str)
    parser.add_argument("--result_dir", help="root directory to save models to", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    BATCH_SIZE = 32
    # training ratio between discriminator and generator
    TRAINING_RATIO = 10
    GRADIENT_PENALTY_WEIGHT = 10
    LR = 0.0001
    BETA_1 = 0.5
    BETA_2 = 0.99

    args = parse_args()
    data_source = args.data
    ROOT_PATH = args.result_dir
    NUM_EPOCHS = args.epochs

    if not os.path.isdir(ROOT_PATH):
        os.makedirs(ROOT_PATH)

    X_train = load_my_data(data_source)
    train(X_train, {
        'generator_save_path': os.path.join(ROOT_PATH, 'generator'),
        'discriminator_save_path': os.path.join(ROOT_PATH, 'discriminator'),
    })
