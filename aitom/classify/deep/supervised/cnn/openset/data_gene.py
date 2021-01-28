import os
import numpy as np
import numpy as N
import pickle


def gene_train_and_test(args):
    # loading data for training.
    # '/scratch/shared_data/xiangruz/classification/domain_adaptation_simulated_/snr-00_5.pickle'
    address = args.raw_data

    temp = np.load(address)
    datai = temp['data'].reshape(args.num_class * args.num_sample_per_class, args.image_size,
                                 args.image_size, args.image_size)
    labelsi = temp['label'].reshape(args.num_class * args.num_sample_per_class)
    # data = data - np.mean(data, axis = 0)
    # data = data / np.std(data, axis = 0)
    # generating the training samples.
    for i in range(args.num_class):
        data_class = datai[args.num_sample_per_class * i:args.num_sample_per_class * i +
                           args.num_sample_per_class, :, :, :]
        if i < args.num_class_train:
            for sub in range(len(data_class)):
                address = args.train_data_dir + str(i)
                if not os.path.exists(address):
                    os.makedirs(address)
                np.save(args.train_data_dir + str(i) + '/' + str(sub) + '.npy', data_class[sub])
        else:
            for sub in range(len(data_class)):
                address = args.test_data_dir + str(i)
                if not os.path.exists(address):
                    os.makedirs(address)
                np.save(args.test_data_dir + str(i) + '/' + str(sub) + '.npy', data_class[sub])


if __name__ == '__main__':
    gene_train_and_test()
