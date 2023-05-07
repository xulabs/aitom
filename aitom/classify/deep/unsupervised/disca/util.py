import sys, multiprocessing, importlib, pickle
from multiprocessing.pool import Pool


import numpy as np
import scipy
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA                                     

import keras
from keras.layers import Input, Dense


def pickle_load(path): 
    """                                                                                                                                                                            
    load a pickle file given path.                                                                                                                      
    """   
    with open(path, 'rb') as f:     o = pickle.load(f, encoding='latin1') 

    return o 



def pickle_dump(o, path, protocol=2):
    """                                                                                                                                                                            
    write a pickle file given the object o and the path.                                                                                                                      
    """ 
    with open(path, 'wb') as f:    pickle.dump(o, f, protocol=protocol)




def run_iterator(tasks, worker_num=multiprocessing.cpu_count(), verbose=False):
    """
    parallel multiprocessing for a given task, this is useful for speeding up the data augmentation step.
    """

    if verbose:		print('parallel_multiprocessing()', 'start', time.time())

    worker_num = min(worker_num, multiprocessing.cpu_count())

    for i,t in tasks.items():
        if 'args' not in t:     t['args'] = ()
        if 'kwargs' not in t:     t['kwargs'] = {}
        if 'id' not in t:   t['id'] = i
        assert t['id'] == i

    completed_count = 0 
    if worker_num > 1:

        pool = Pool(processes = worker_num)
        pool_apply = []
        for i,t in tasks.items():
            aa = pool.apply_async(func=call_func, kwds={'t':t})

            pool_apply.append(aa)


        for pa in pool_apply:
            yield pa.get(99999)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()

        pool.close()
        pool.join()
        del pool

    else:

        for i,t in tasks.items():
            yield call_func(t)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()
	
    if verbose:		print('parallel_multiprocessing()', 'end', time.time())

    
run_batch = run_iterator #alias



def call_func(t):

    if 'func' in t:
        assert 'module' not in t
        assert 'method' not in t
        func = t['func']
    else:
        modu = importlib.import_module(t['module'])
        func = getattr(modu, t['method'])

    r = func(*t['args'], **t['kwargs'])
    return {'id':t['id'], 'result':r}



def random_rotation_matrix():
    """
    generate a random 3D rigid rotation matrix.
    """
    m = np.random.random( (3,3) )
    u,s,v = np.linalg.svd(m)

    return u


def rotate3d_zyz(data, Inv_R, center=None, order=2):
    """
    rotate a 3D data using ZYZ convention (phi: z1, the: x, psi: z2).
    """
    # Figure out the rotation center
    if center is None:
        cx = data.shape[0] / 2
        cy = data.shape[1] / 2
        cz = data.shape[2] / 2
    else:
        assert len(center) == 3
        (cx, cy, cz) = center

    
    from scipy import mgrid
    grid = mgrid[-cx:data.shape[0]-cx, -cy:data.shape[1]-cy, -cz:data.shape[2]-cz]
    temp = grid.reshape((3, np.int(grid.size / 3)))
    temp = np.dot(Inv_R, temp)
    grid = np.reshape(temp, grid.shape)
    grid[0] += cx
    grid[1] += cy
    grid[2] += cz

    # Interpolation
    from scipy.ndimage import map_coordinates
    d = map_coordinates(data, grid, order=order)

    return d



def data_augmentation(x_train, factor = 2):
    """
    data augmentation given the training subtomogram data.
    if factor = 1, this function will return the unaugmented subtomogram data.
    if factor > 1, this function will return (factor - 1) number of copies of augmented subtomogram data.
    """

    if factor > 1:

        x_train_augmented = []
        
        x_train_augmented.append(x_train)

        for f in range(1, factor):
            ts = {}        
            for i in range(len(x_train)):                       
                t = {}                                                
                t['func'] = rotate3d_zyz                                   
                                                      
                # prepare keyword arguments                                                                                                               
                args_t = {}                                                                                                                               
                args_t['data'] = x_train[i,:,:,:,0]                                                                                                                    
                args_t['Inv_R'] = random_rotation_matrix()                                                   
                                                                                                                                                                                                                                           
                t['kwargs'] = args_t                                                  
                ts[i] = t                                                       
                                                                      
            rs = run_batch(ts, worker_num=48)
            x_train_f = np.expand_dims(np.array([_['result'] for _ in rs]), -1)
        
            x_train_augmented.append(x_train_f)
            
        x_train_augmented = np.concatenate(x_train_augmented)
    
    else:
        x_train_augmented = x_train                        

        x_train[x_train == 0] = np.random.normal(loc=0.0, scale=1.0, size = np.sum(x_train == 0))

    return x_train_augmented



def one_hot(a, num_classes):
    """
    one-hot encoding. 
    """
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])   



def smooth_labels(labels, factor=0.1):
    """
    label smoothing. 
    """                                                                                                                                                     
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])

    return labels 


def remove_empty_cluster(labels):
    """
    if there are no samples in a cluster,
    this function will remove the cluster and make the remaining cluster number compact. 
    """
    labels_unique = np.unique(labels)
    for i in range(len(np.unique(labels))):
        labels[labels == labels_unique[i]] = i

    return labels
                        


def YOPO_classification(num_labels, vector_size = 1024):
    """
    last classification layer of YOPO.  
    """

    input_shape = (vector_size,)                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    main_input = Input(shape= input_shape, name='main_input')
    
    m = Dense(num_labels, activation='softmax')(main_input)

    mod = keras.models.Model(inputs=main_input, outputs=m)                                     
                                                                               
    return mod 



def statistical_fitting(features, labels, candidateKs, K, reg_covar, i):
    """
    fitting a Gaussian mixture model to the extracted features from YOPO
    given current estimated labels, K, and a number of candidateKs. 

    reg_covar: non-negative regularization added to the diagonal of covariance.     
    i: random_state for initializing the parameters.
    """

    pca = PCA(n_components=16)  
    features_pca = pca.fit_transform(features) 

    labels_K = [] 
    BICs = [] 
                                                                                                                                                            
    for k in candidateKs: 
        if k == K: 
            try:
                weights_init = np.array([np.sum(labels == j)/np.float(len(labels)) for j in range(k)]) 
                means_init = np.array([np.mean(features_pca[labels == j], 0) for j in range(k)]) 
                precisions_init = np.array([np.linalg.inv(np.cov(features_pca[labels == j].T) + reg_covar * np.eye(features_pca.shape[1])) for j in range(k)]) 
 
                gmm_0 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i,  
                                        weights_init=weights_init, means_init=means_init, precisions_init=precisions_init, init_params = 'random') 
 
                gmm_0.fit(features_pca) 
                labels_k_0 = gmm_0.predict(features_pca)

            except:     
                gmm_0 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i, init_params = 'random') 
                gmm_0.fit(features_pca) 
                labels_k_0 = gmm_0.predict(features_pca) 
                         
         
            gmm_1 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i, init_params = 'random') 
            gmm_1.fit(features_pca) 
            labels_k_1 = gmm_1.predict(features_pca) 
             
            m_select = np.argmin([gmm_0.bic(features_pca), gmm_1.bic(features_pca)]) 
             
            if m_select == 0: 
                labels_K.append(labels_k_0) 
                 
                BICs.append(gmm_0.bic(features_pca)) 
             
            else: 
                labels_K.append(labels_k_1) 
                 
                BICs.append(gmm_1.bic(features_pca)) 
         
        else: 
            gmm = GaussianMixture(n_components=k, covariance_type='full', tol=0.0001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i, init_params = 'random') 
         
            gmm.fit(features_pca) 
            labels_k = gmm.predict(features_pca) 

            labels_K.append(labels_k) 
             
            BICs.append(gmm.bic(features_pca)) 
    
    labels_temp = remove_empty_cluster(labels_K[np.argmin(BICs)])                     
     
    K_temp = len(np.unique(labels_temp)) 
     
    if K_temp == K: 
        same_K = True 
    else: 
        same_K = False 
        K = K_temp     

    print('Estimated K:', K)
    
    return labels_temp, K, same_K, features_pca         



def update_output_layer(K, label_one_hot, batch_size, model_feature, features, lr):
    """
    this function updates the output classification layer when the estimated number of clusters change.
    the new output classification layer will be tuned given the current extracted features and estimated labels.
    """

    model_classification = YOPO_classification(num_labels=K) 

    optimizer = keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
     
    model_classification.compile(optimizer= optimizer, loss='categorical_hinge',  metrics=['accuracy']) 
         
    model_classification.fit(features, label_one_hot, epochs=10, batch_size=batch_size, shuffle=True, verbose = 0) 
     
    ### New YOPO ### 
    model = keras.models.Model(model_feature.layers[0].get_input_at(0), model_classification(model_feature.layers[-1].get_output_at(0))) 
    optimizer = keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 

    model.compile(optimizer= optimizer, loss='categorical_hinge',  metrics=['accuracy'])

    return model



def prepare_training_data(x_train, labels, label_smoothing_factor):
    """
    training data preparation given the current training data x_train, labels, and label_smoothing_factor
    """

    label_one_hot = one_hot(labels, len(np.unique(labels))) 
     
    index = np.array(range(x_train.shape[0] * 2)) 

    np.random.shuffle(index)         
     
    x_train_augmented = data_augmentation(x_train, 2) 
     
    x_train_permute = x_train_augmented[index].copy() 
     
    label_smoothing_factor *= 0.9 
     
    labels_augmented = np.tile(smooth_labels(label_one_hot, label_smoothing_factor), (2,1))               
     
    labels_permute = labels_augmented[index].copy() 

    return label_one_hot, x_train_permute, label_smoothing_factor, labels_permute
    


