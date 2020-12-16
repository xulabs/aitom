import os
import numpy as np
import random


def gene_train_list(args):
    filename = args.train_list_dir
    address = args.train_data_dir

    for i in os.listdir(address):
        temp = os.listdir(str(os.path.join(address, i)))
        for j in temp:
            final = os.path.join(str(os.path.join(address, i)), j)
            with open(filename, 'a') as fileobject:
                fileobject.writelines(str(final) + '\n')
    fileobject.close()


def gene_verification_pair(args):
    filename = args.test_list_dir
    # generate the paired images.
    for item in os.listdir(args.test_data_dir):
        data = list(os.listdir(str(os.path.join(args.test_data_dir, item))))
        # print(len(data))
        for num in range(100):
            pair = random.sample(data, 2)
            for i in range(len(pair)):
                pair[i] = pair[i][:-4]

            with open(filename, 'a') as fileobject:
                fileobject.writelines(str(item) + ' ' + str(pair[0]) + ' ' + str(pair[1]) + '\n')

    # generate unpaired images.
    profile = list(range(args.num_class_train, args.num_class))
    for item in os.listdir(args.test_data_dir):
        data = list(os.listdir(str(os.path.join(args.test_data_dir, item))))
        # print (len(data))
        for num in range(200):
            pair = random.sample(data, 2)
            pair_2 = random.sample(profile, 2)
            for i in range(len(pair)):
                pair[i] = pair[i][:-4]

            with open(filename, 'a') as fileobject:
                fileobject.writelines(
                    str(pair_2[0]) + ' ' + str(pair[0]) + ' ' + str(pair_2[1]) + ' ' +
                    str(pair[1]) + '\n')


if __name__ == '__main__':
    gene_verification_pair()
