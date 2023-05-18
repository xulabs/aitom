import keras
from keras.layers import Input, Dense, Conv3D, Activation, GlobalMaxPooling3D, GaussianDropout, BatchNormalization, Concatenate, ELU

import numpy as np
import scipy
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix 
from scipy.optimize import linear_sum_assignment

from aitom.classify.deep.unsupervised.disca.util import *



                                                                                                                                                                                                        
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
    """                                                                                                                                                                            
    compute the Distortion-based Davies-Bouldin index defined in Equ 1 of the Supporting Information.                                                                                                        
    """ 

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
    """                                                                                                                                                                            
    feature extraction part of the YOPO network                                                                                                        
    """

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

    mod = keras.models.Model(inputs=main_input, outputs=out)                                     
                                                                               
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


    

if __name__ == '__main__':
    ### Define Parameters Here ###
    
    image_size = 24   ### subtomogram size ### 
    candidateKs = [5,10,20]   ### candidate number of clusters to test, it is also possible to set just one large K that overpartites the data  
          
    batch_size = 64      
    M = 20   ### number of iterations ### 
    lr = 1e-5   ### CNN learning rate ### 

    label_smoothing_factor = 0.2   ### label smoothing factor ### 
    reg_covar = 0.00001 


    model_path = '.h5'   ### path for saving keras model, should be a .h5 file ### 
    label_path = '.pickle' ### path for saving labels, should be a .pickle file ###
 
    ### Load Dataset Here ###     
     
    x_train =   ### load the x_train data, should be shape (n, shape_1, shape_2, shape_3, 1)

    gt =   ### load or define label ground truth here, if for simulated data 


    ### Generalized EM Process ### 
    K = None 
    labels = None
    DBI_best = np.float('inf') 

    done = False 
    i = 0 
     
    while not done: 
        print('Iteration:', i) 
         
    ### Feature Extraction ### 
        if i == 0: 
            model_feature = YOPO_feature(image_size)   ### create a new nodel
 
        else:                
            model_feature = keras.models.Model(inputs=model.get_layer('input_1').input, outputs=model.get_layer('fc2').output) 
 
        model_feature.compile(loss='categorical_hinge', optimizer='adam')                 
         
        features = model_feature.predict(x_train)          
 
         
    ### Feature Clustering ###                              
             
        labels_temp, K, same_K, features_pca = statistical_fitting(features = features, labels = labels, candidateKs = candidateKs, K = K, reg_covar = reg_covar, i = i) 
         
    ### Matching Clusters by Hungarian Algorithm ### 
        if same_K: 
            labels_temp = align_cluster_index(labels, labels_temp) 

        i, labels, done = convergence_check(i = i, M = M, labels_temp = labels_temp, labels = labels, done = done) 
             
        print('Cluster sizes:', [np.sum(labels == k) for  k in range(K)])         
 
    ### Validate Clustering by distortion-based DBI ###             
        DBI = DDBI(features_pca, labels) 
 
        if DBI < DBI_best: 
            if i > 1:             
               model.save(model_path)   ### save model here ### 
                 
               pickle_dump(labels, label_path) 
 
            labels_best = labels   ### save current labels if DDBI improves ###  
 
            DBI_best = DBI     
                             
        print('DDBI:', DBI, '############################################') 

#        if K == 5:   ### This is for evaluating accuracy on simulated data        
#            labels_gt = align_cluster_index(gt, labels) 
 
#            print('Accuracy:', np.sum(labels_gt == gt), '############################################') 
 
#        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(gt, labels)                                                             
                                                      
#        print('Homogeneity score:', homogeneity, '############################################')                                                          
#        print('Completeness score:', completeness, '############################################')                                              
#        print('V_measure:', v_measure, '############################################')                                                                    


    ### Permute Samples ###             
    
        label_one_hot, x_train_permute, label_smoothing_factor, labels_permute = prepare_training_data(x_train = x_train, labels = labels, label_smoothing_factor = label_smoothing_factor)
     
         
    ### Finetune new model with current estimated K ### 
        if not same_K: 
            model = update_output_layer(K = K, label_one_hot = label_one_hot, batch_size = batch_size, model_feature = model_feature, features = features, lr = lr)
 
             
    ### CNN Training ###           
 
        lr *= 0.95 
        keras.backend.set_value(model.optimizer.lr, lr)         ### Decay Learning Rate ###                 

        model.fit(x_train_permute, labels_permute, epochs=1, batch_size=batch_size, shuffle=True) 
            

