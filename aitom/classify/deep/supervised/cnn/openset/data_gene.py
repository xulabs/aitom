import os
import numpy as np
import numpy as N
import pickle

#loading data for training. 
address =  '/scratch/shared_data/xiangruz/classification/domain_adaptation_simulated_/snr-00_5.pickle'
a = open(address, 'rb')
data = pickle.load(a, encoding = 'latin1')

#data = data - np.mean(data, axis = 0)
#data = data / np.std(data, axis = 0) 


def vol_to_image_stack(vs): #generate the 
    image_size = vs[0].shape[0]
    sample_size = len(vs)
    num_channels=1
    sample_data = N.zeros((sample_size, image_size, image_size, image_size, num_channels), dtype=N.float32)#1000
    for i,v in enumerate(vs):        
        sample_data[i, :, :, :, 0] = v
    return sample_data

def pdb_id_label_map(pdb_ids):
    pdb_ids = set(pdb_ids)
    pdb_ids = list(pdb_ids)
    m = {p:i for i,p in enumerate(pdb_ids)}
    return m # the dictionary

def list_to_data(dj, pdb_id_map=None):
    re = {}
    re['data'] = vol_to_image_stack(vs=[_['v'] for _ in dj])#numpy array
    if pdb_id_map is not None:
        labels = N.array([pdb_id_map[_['pdb_id']] for _ in dj])
        #print(labels)
        from keras.utils import np_utils
        #labels = np_utils.to_categorical(labels, len(pdb_id_map))
        re['labels'] = labels
    return re,  N.array([pdb_id_map[_['pdb_id']] for _ in dj])
    
pdb_id_map_i = pdb_id_label_map([_['pdb_id'] for _ in data])
dati, scalar_label = list_to_data(data, pdb_id_map_i)
datai = dati['data'].reshape(23000,40,40,40)
labelsi = dati['labels']


print(labelsi[999])
print(labelsi[998])
print(labelsi[1001])
print(labelsi[1000])

'''
pdb_id_map_i = {0:'proteasome_d', 1:'none', 2:'TRiC', 3:'ribosome', 4:'proteasome_s', 5:'membrane'}
for i in range(6):
    data_class = data[pdb_id_map_i[i]]
    if i < 3:
        for sub in range(len(data_class)):
            address = './qiang_train/'+str(i)#+'/'+str(sub)+'.npy'
            if not os.path.exists(address):
                os.makedirs(address)
            np.save('./qiang_train/'+str(i)+'/'+str(sub)+'.npy', data_class[sub])

    else:
        for sub in range(len(data_class)):
            address = './qiang_test/'+str(i)#+'/'+str(sub)+'.npy'
            if not os.path.exists(address):
                os.makedirs(address)
            np.save('./qiang_test/'+str(i)+'/'+str(sub)+'.npy', data_class[sub])
'''


#generating the training samples. 
for i in range(23):
	data_class = datai[1000*i:1000*i+1000, :, :, :]
	if i < 17:
		for sub in range(len(data_class)):
			address = './0.5_train/'+str(i)
			if not os.path.exists(address):
				os.makedirs(address)
			np.save('./0.5_train/'+str(i)+'/'+str(sub)+'.npy', data_class[sub])
	else:
		for sub in range(len(data_class)):
			address = './0.5_test/'+str(i)
			if not os.path.exists(address):
				os.makedirs(address)
			np.save('./0.5_test/'+str(i)+'/'+str(sub)+'.npy', data_class[sub])
