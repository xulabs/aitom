import time, pickle

import keras
from keras.layers import Input, Dense, Conv3D, Activation, GlobalMaxPooling3D, GaussianDropout, BatchNormalization, Concatenate, ELU

import numpy as np
import scipy
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix 
from scipy.optimize import linear_sum_assignment

from aitom_core.classify.deep.unsupervised.disca.training import *


def pickle_load(path): 
    with open(path, 'rb') as f:     o = pickle.load(f, encoding='latin1') 

    return o 



def pickle_dump(o, path, protocol=2):
    with open(path, 'wb') as f:    pickle.dump(o, f, protocol=protocol)


                                                                                                                                                                                                        
def align_cluster_index(ref_cluster, map_cluster):                                                                                                                                        
    """                                                                                                                                                                            
    remap cluster index according the the ref_cluster.                                                                                                                                    
    both inputs must have same number of unique cluster index values.                                                                                                                      
    """                                                                                                                                                                                   
                                                                                                                                                                                   
    ref_values = np.unique(ref_cluster)                                                                                                                                                   
    map_values = np.unique(map_cluster)                                                                                                                                                   
                                                                                                                                                                                                        
    if ref_values.shape[0]!=map_values.shape[0]:                                                                                                                                   
        print('error: both inputs must have same number of unique cluster index values.')                                                                                                      
        return()                                                                                                                                                               
    cont_mat = contingency_matrix(ref_cluster, map_cluster)                                                                                                                 
                                                                                                                                                                                                        
    row_ind, col_ind = linear_sum_assignment(len(ref_cluster) - cont_mat)                                                                                            
                                                                                                                                                                                                        
    map_cluster_out = map_cluster.copy()                                                                                                                                           
                                                                                                                                                                                                        
    for i in ref_values:                                                                                                                                                            
                                                                                                                                                                                                        
        map_cluster_out[map_cluster == col_ind[i]] = i                                                                                                                                     

    return map_cluster_out                                                                                                                                                                


    
def DDBI(features, labels):

    means_init = np.array([np.mean(features[labels == i], 0) for i in np.unique(labels)])
    precisions_init = np.array([np.linalg.inv(np.cov(features[labels == i].T) + reg_covar * np.eye(features.shape[1])) for i in np.unique(labels)])

    T = np.array([np.mean(np.diag((features[labels == i] - means_init[i]).dot(precisions_init[i]).dot((features[labels == i] - means_init[i]).T))) for i in np.unique(labels)])
    
    D = np.array([np.diag((means_init - means_init[i]).dot(precisions_init[i]).dot((means_init - means_init[i]).T)) for i in np.unique(labels)])
    
    DBI_matrix = np.zeros((len(np.unique(labels)), len(np.unique(labels))))
    
    for i in range(len(np.unique(labels))):
        for j in range(len(np.unique(labels))):
            if i != j:
                DBI_matrix[i, j] = (T[i] + T[j])/(D[i, j] + D[j, i])
            
    DBI = np.mean(np.max(DBI_matrix, 0))
    

    return DBI                        


                                                                                                          
def YOPO_feature(image_size):
    kernel_initializer = keras.initializers.orthogonal()
    bias_initializer = keras.initializers.zeros()


    input_shape = (image_size, image_size, image_size, 1)                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    main_input = Input(shape= input_shape, name='input_1')
    
    x = GaussianDropout(0.5)(main_input)                                                          

    x = Conv3D(64, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m1 = GlobalMaxPooling3D()(x)                                                                                                                                             

    x = Conv3D(80, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m2 = GlobalMaxPooling3D()(x)                                                                                                                                             

    x = Conv3D(96, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m3 = GlobalMaxPooling3D()(x)                                                                                                                                             

    x = Conv3D(112, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m4 = GlobalMaxPooling3D()(x)                                                                                                                                             

    x = Conv3D(128, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m5 = GlobalMaxPooling3D()(x)                                                                                                                                             

    x = Conv3D(144, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m6 = GlobalMaxPooling3D()(x)                                                                                                                                             

    x = Conv3D(160, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m7 = GlobalMaxPooling3D()(x)                                                                                                                                             

    x = Conv3D(176, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m8 = GlobalMaxPooling3D()(x)                                                                                                                                             

    x = Conv3D(192, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m9 = GlobalMaxPooling3D()(x)                                                                                                                                             

    x = Conv3D(208, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = ELU()(x)                         
    x = BatchNormalization()(x)
    m10 = GlobalMaxPooling3D()(x)                                                                                                                                             
                                                                                                                                                                                                                                                                                      
    m = Concatenate()([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10])
    m = BatchNormalization()(m)
                                                                                  

    out = Dense(1024, name='fc2', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(m)    

    mod = keras.models.Model(input=main_input, output=out)                                     
                                                                               
    return mod                                                                 



def convergence_check(i, M, labels_temp, labels, done):

    if i > 0:
        if np.sum(labels_temp == labels)/np.float(len(labels)) > 0.999: 
            done = True 

    i += 1 
    if i == M: 
        done = True
        
    labels = labels_temp
    
    return i, labels, done                 
