import sys
sys.path.append('./models/')

import os
import numpy as np
from PIL import Image
from iw_models import w_generator_model, w_discriminator_model, regressor_model, large_regressor_model, large_generator_model, large_discriminator_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

''' 
Functions to make regressor, generator, discriminator - derived from iw_models
'''

GEN_INPUT_DIM = 100
IMG_DIM = 64


def make_regressor(discriminator):
    return large_regressor_model((IMG_DIM, IMG_DIM, IMG_DIM, 1), GEN_INPUT_DIM, discriminator)

def make_generator():
    return large_generator_model(GEN_INPUT_DIM)

def make_discriminator():
    return large_discriminator_model((IMG_DIM, IMG_DIM, IMG_DIM, 1))

# almost equal within epsilon
def almost_equal(a,b,epsilon=0.5):
    return abs(a-b) < epsilon

# normalize array a to be between [lo, hi]
def make_between_values(a, lo, hi):
    a = (a - a.min()) / (a.max() - a.min())
    a = a * (hi - lo) + lo
    return a

# normalize array s to be between [-1, 1]
def normalize(s):
    return make_between_values(s, -1., 1.)

# gets random vector to input into generator
def get_random_vector(num):
    res = np.random.normal(loc=0.0, scale=1.0, size=(num, GEN_INPUT_DIM))
    return res.astype(np.float32)

# normalizes shape [-1, 1] (tanh as last layer of generator), and makes it right shape
def inputToRegressor(myShapes):
    if (len(myShapes.shape) <= 3): # single shape as input
        myShapes = [myShapes]
    normalized = map(lambda s: 2. * (s - s.min()) / (s.max() - s.min()) - 1, myShapes)
    return np.reshape(normalized, (-1, IMG_DIM, IMG_DIM, IMG_DIM, 1))


# convert a 3D cube to a 2D image of slices
def cub_img(v, view_dir=1):
    import numpy as N
    v = N.reshape(v, (IMG_DIM, IMG_DIM, IMG_DIM))
    if view_dir == 0:
        vt = N.transpose(v, [1,2,0])
    elif view_dir == 1:
        vt = N.transpose(v, [2,0,1])
    elif view_dir == 2:
        vt = v
    
    row_num = vt.shape[0] + 1
    col_num = vt.shape[1] + 1
    slide_num = vt.shape[2]
    disp_len = int( N.ceil(N.sqrt(slide_num)) )
    
    slide_count = 0
    im = N.zeros( (row_num*disp_len, col_num*disp_len) ) + float('nan')
    for i in range(disp_len):
        for j in range(disp_len):
            im[(i*row_num) : ((i+1)*row_num-1),  (j*col_num) : ((j+1)*col_num-1)] = vt[:,:, slide_count]
            slide_count += 1
            
            if (slide_count >= slide_num):
                break
            
        
        if (slide_count >= slide_num):
            break
   
    im_v = im[N.isfinite(im)]

    # normalizing between 0 and 1
    try:
        if im_v.max() > im_v.min(): 
            im = (im - im_v.min()) / (im_v.max() - im_v.min())
    except:
        import pdb; pdb.set_trace()

    return {'im':im, 'vt':vt}


def makeParentDirectories(path):
    parent = "/".join(path.split("/")[:-1])
    if not os.path.isdir(parent):
        os.makedirs(parent)

'''
Puts shapes slices side by side
'''
def saveSlices(shapes, save_path):
    (s1,s2) = cub_img(shapes[0])['im'].shape
    space = 10
    x_offset = 0
    im = Image.new('L', (len(shapes) * s1 + (len(shapes)-1) * space, s2))

    for shape in shapes:
        img = cub_img(shape)['im'] * 256.
        im.paste(Image.fromarray(img.astype(np.uint8)), (x_offset, 0))
        x_offset += s1+space

    makeParentDirectories(save_path)
    im.save(save_path)


def plotLosses(plot_dict, params_dict, path_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    legends = sorted(plot_dict.keys())
    for k in legends:
        ax.plot(plot_dict[k])

    ax.set_title(params_dict['title'])
    ax.set_xlabel(params_dict['xlabel'])
    ax.set_ylabel(params_dict['ylabel'])
    ax.legend(legends, loc='upper left')

    makeParentDirectories(path_name)
    fig.savefig(path_name)