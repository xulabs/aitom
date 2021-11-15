import numpy as np
from keras.models import load_model
import keras.backend as K
import pickle

from aitom.align.deep.gum.models.Gum_Net import *
from aitom.align.deep.gum.opt import opt

def main(opt):
    build_model = opt.build_model
    initial_lr = opt.initial_lr

    # 1. load demo data
    with open(opt.data_path, 'rb') as f:
        x_test, y_test, observed_mask, missing_mask, ground_truth = pickle.load(
            f, encoding='latin1')
    image_size = x_test[0].shape[0]

    # 2. load or build model
    if not build_model:
        model = load_model(opt.model_path,
                           custom_objects={
                               'SpectralPooling': SpectralPooling,
                               'RigidTransformation3DImputation': RigidTransformation3DImputation,
                               'FeatureCorrelation': FeatureCorrelation,
                               'FeatureL2Norm': FeatureL2Norm,
                               'correlation_coefficient_loss':correlation_coefficient_loss})
    else:
        model = GUM([32, 32, 32])

    K.set_value(model.optimizer.lr, initial_lr)

    # 3. normalize data for better model adaptation
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    y_test = (y_test - np.mean(y_test)) / np.std(y_test)

    # 4. save a copy for evaluation
    x_test_copy = x_test.copy()
    y_test_copy = y_test.copy()

    # 5. generate imputation masks per sample
    mask_1 = np.tile(np.expand_dims(observed_mask, -1),
                     (x_test.shape[0], 1, 1, 1, 1))
    mask_2 = np.tile(np.expand_dims(missing_mask, -1),
                     (x_test.shape[0], 1, 1, 1, 1))

    # 6. predict 6D transformation parameters
    transformation_output = get_transformation_output_from_model(
        model, x_test_copy, y_test_copy, mask_1, mask_2)

    # 7. compare with ground truth
    print('Before finetuning:')
    alignment_eval(ground_truth, transformation_output, image_size)

    # 8. finetune the model for 20 iterations
    for i in range(20):
        print('Training Iteration ' + str(i))
        K.set_value(model.optimizer.lr, initial_lr * 0.9 ** i)

        # each iteration gets random pairs of subtomograms
        np.random.shuffle(x_test)
        np.random.shuffle(y_test)

        model.fit([x_test, y_test, mask_1, mask_2],
                  y_test,
                  epochs=1,
                  batch_size=32)

    # 9. predict the 6D transformation parameters
    transformation_output = get_transformation_output_from_model(
        model, x_test_copy, y_test_copy, mask_1, mask_2)

    # 10. compare with ground truth
    print('After finetuning:')
    alignment_eval(ground_truth, transformation_output, image_size)

if __name__ == '__main__':
    main(opt)

    
