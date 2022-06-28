import numpy as np
import os
import pickle
import json

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def reconstrate(data, final_size, num):
    piecesize = data.shape[-1]
    # num = int(pow(data.shape[0], 1/3))
    print(data.shape[0])
    print(num)
    repeat = int((num*piecesize-final_size)/(num-1))

    output = np.zeros((final_size,final_size,final_size))
    for x in range(num):
        for y in range(num):
            for z in range(num):
                tube = data[z+y*num+x*num*num]
                output[x*(piecesize-repeat):x*(piecesize-repeat)+piecesize,
                       y*(piecesize-repeat):y*(piecesize-repeat)+piecesize, z*(piecesize-repeat):z*(piecesize-repeat)+piecesize] = tube
    
    return output


if __name__ == '__main__':
    root = "./result/wgan4_sim_{}.pickle"

    for label, size, num in [(5, 250, 7), (400,600, 15)]:
        data = read_pickle(root.format(label))
        data = np.array(data)
        print(data.shape)
        output = reconstrate(data, size, num)

        np.save('./result/{}_bigmap.npy'.format(label), output)
