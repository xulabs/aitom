import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def get_initial_weights(output_size):
    b = np.random.random((6,)) - 0.5
    W = np.zeros((output_size, 6), dtype = 'float32')
    weights = [W, b.flatten()]
    return weights    


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true                      
    y = y_pred                                  
    mx = K.mean(x)  

    my = K.mean(y)                                     
    xm, ym = x-mx, y-my                                                
    r_num = K.sum(tf.multiply(xm,ym))                                     
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)

    return 1 - K.square(r)
