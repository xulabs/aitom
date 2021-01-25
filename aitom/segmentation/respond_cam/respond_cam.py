'''
Author: Guanan Zhao
'''


import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import keras


def grad_cam(cnn_model, data, class_index, target_layer_name, scores_symbol=None, dim=3):
    return x_cam(cnn_model, data, class_index, target_layer_name, respond=False,
      scores_symbol=scores_symbol, dim=dim)

def respond_cam(cnn_model, data, class_index, target_layer_name, scores_symbol=None, dim=3):
    return x_cam(cnn_model, data, class_index, target_layer_name, respond=True,
      scores_symbol=scores_symbol, dim=dim)

# The common parts of Grad-CAM and Respond-CAM
#  scores_symbol is where the variable of class score is.
#  By default (for our CNN-1 and CNN-2), it is the input of the softmax_layer.
def x_cam(cnn_model, data, class_index, target_layer_name, respond, scores_symbol=None, dim=3):
    # Get the variable of class score
    if scores_symbol is None:
        softmax_layer, = [l for l in cnn_model.layers if l.name == 'softmax']
        scores_symbol = softmax_layer.input
    class_count = cnn_model.weights[-1].shape.dims[-1].value # i.e. length of the last bias vector
    class_score = K.sum(K.dot(scores_symbol, K.transpose(K.one_hot([class_index], class_count))))

    # Get the variable of target layer output
    target_layer, = [l for l in cnn_model.layers if l.name == target_layer_name]
    activation_symbol = target_layer.output

    # Define the calculation function
    gradient_symbol, = K.gradients(class_score, activation_symbol)
    func = K.function(
      [cnn_model.layers[0].input, K.learning_phase()],
      [activation_symbol, gradient_symbol]
    )

    # Get the values of activation and gradient:
    if dim == 3:
        data_expanded = np.expand_dims([data], axis=-1)
    else:
        data_expanded = data
    activation, gradient = func([data_expanded, 0])
    activation, gradient = activation[0], gradient[0]

    # Get the CAM:
    axis = tuple(range(dim)) # (0,1,2) for 3D and (0,1) for 2D
    if respond:
        weights = np.sum(activation * gradient, axis=axis) \
          / (np.sum(activation + 1e-10, axis=axis))
    else:
        weights = np.mean(gradient, axis=axis)
    cam = np.sum(activation * weights, axis=-1)
    
    return cam


# The function used for the experiment on the "sum-to-score property"
def get_all_scores_and_camsums(cnn_model, target_layer_name, dj):
    # Define the calculation function
    softmax_layer, = [l for l in cnn_model.layers if l.name == 'softmax']
    scores_symbol = softmax_layer.input
    class_count = cnn_model.weights[-1].shape.dims[-1].value
    class_scores = []
    for i in range(class_count):
        class_scores.append(K.sum(K.dot(scores_symbol, K.transpose(K.one_hot([i], class_count)))))
    target_layer, = [l for l in cnn_model.layers if l.name == target_layer_name]
    activation_symbol = target_layer.output
    gradients_symbols = []
    for c in class_scores:
        gradients_symbols += K.gradients(c, activation_symbol)

    func = K.function(
      [cnn_model.layers[0].input, K.learning_phase()], 
      gradients_symbols + [activation_symbol, scores_symbol]
    )

    # Apply the function for each data image in dj
    camsums_grad = []
    camsums_respond = []
    scores = []
    for i, d in enumerate(dj):
        print('\x1b[2K%d / %d\x1b[1A' % (i, len(dj)))
        data = d['v']
        data_expanded = np.expand_dims([data], axis=-1)
        values = func([data_expanded, 0])
        gradients = np.array(values[0:class_count])[:,0,:,:,:,:]
        activation = values[-2][0,:,:,:,:]
        score = values[-1][0,:]
        camsum_grad = []
        camsum_respond = []

        for class_index in range(class_count):
            grad_weights = np.mean(gradients[class_index], axis=(0,1,2))
            grad_cam = np.sum(activation * grad_weights, axis=-1)
            camsum_grad.append(np.sum(grad_cam))

            respond_weights = np.sum(activation * gradients[class_index], axis=(0,1,2)) \
              / (np.sum(activation + 1e-10, axis=(0,1,2)))
            respond_cam = np.sum(activation * respond_weights, axis=-1)
            camsum_respond.append(np.sum(respond_cam))

        camsums_grad.append(np.array(camsum_grad))
        camsums_respond.append(np.array(camsum_respond))
        scores.append(score)
    return camsums_grad, camsums_respond, scores
