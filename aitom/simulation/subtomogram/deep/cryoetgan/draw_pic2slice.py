import numpy as N
import png          # in pypng package
import os
#import cPickle as pickle
import pickle

def save_png(m, name, normalize=True, verbose=False):

    if verbose:
        print('save_png()')
        print('unique values', sorted(set(m.flatten())))

    m = N.array(m, dtype=N.float)

    mv = m[N.isfinite(m)]
    if normalize:
        # normalize intensity to 0 to 1
        if mv.max() - mv.min() > 0:
            m = (m - mv.min()) / (mv.max() - mv.min())
        else:
            m = N.zeros(m.shape)
    else:
        assert mv.min() >= 0
        assert mv.max() <= 1

    m = N.ceil(m * 65534)
    m = N.array(m, dtype=N.uint16)

    
    png.from_array(m, 'L').save(name)
    
    
def cub_img(v, view_dir=2):
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

    if im_v.max() > im_v.min(): 
        im = (im - im_v.min()) / (im_v.max() - im_v.min())

    return {'im':im, 'vt':vt}

def uncertainty_est(img_size = 28, draw_num=1, mean_num=5):
    root = '../result/'
    data = '../visual/'
    for kind in ["wgan4_sim_lr2e-5"]:
        if not os.path.exists(data+kind):
            os.makedirs(data+ kind)
        
        with open('../result/{}/uncertainty_est_sub_dropout80p.pickle'.format(kind), "rb") as pic:
            pic_sub = pickle.load(pic, encoding='latin1')

        if img_size == 40:
            id2label = {"ribosome": 0, "membrane": 1, "TRiC": 2, "proteasome_s": 3}
        else:
            id2label = {"31": 0, "33": 1, "35": 2, "43": 3, "69": 4, "72": 5, "73": 6}

        drawingimgs = {}
        
        for category, idx in id2label.items():
            subs = pic_sub[category]
            drawingimgs[category] = []
            for i in range(draw_num):
                drawingimgs[category].append(N.std(N.array(subs[mean_num*i:mean_num*(i+1)]).reshape((-1, img_size,img_size,img_size)), axis=0))
        
        for idx, image in drawingimgs.items():
            image = N.array(image)
            print(idx)
            print(image.shape)
            if not os.path.exists(data+kind+'/uncertainty_'+idx):
                os.makedirs(data+kind+'/uncertainty_'+idx)
            for i in range(image.shape[0]):
                save_png(cub_img(image[i,:].reshape((img_size,img_size,img_size)))['im'], data+kind+'/uncertainty_'+idx+'/'+idx+'_{}_sub.png'.format(i))
        

def main(img_size=28):
    root = '../result/'
    data = '../visual/'
    if img_size == 40:
        id2label = {0:"ribosome", 1:"membrane", 2:"TRiC", 3:"proteasome_s"}
    else:
        id2label = {0:"31", 1:"33", 2:"35", 3:"43", 4:"69", 5:"72", 6:"73"}

    for kind in ["wgan7_sim_lr2e-4"]:
        if not os.path.exists(data+kind):
            os.makedirs(data+ kind)
        # with open(data + kind + "fake_sub.pickle", "rb") as pic:
        #     pic_sub = pickle.load(pic, encoding='latin1')

        # pic_sub = N.load(os.path.join(root, kind+'.npy'))
        # save_png(cub_img(pic_sub)['im'], data+kind+'/big.png')
        
        # for idx, image in pic_sub.items():
        #     image = N.array(image)
        #     for i in range(10):
        #         save_png(cub_img(image[i,:].reshape((28,28,28)))['im'], data+kind+'/'+idx+'_{}fake_sub.png'.format(i))

        # with open("/shared/home/v_xindi_wu/proj/data/density_map_40_5.pickle", "rb") as pic:
        #     pic_sub = pickle.load(pic, encoding='latin1')
        # for idx, image in pic_sub.items():
        #     image = N.array(image)
        #     for i in range(10):
        #         save_png(cub_img(image[i,:].reshape((40,40,40)))['im'], data+kind+'/'+idx+'_{}_real_den.png'.format(i))

        # with open('/shared/home/v_xindi_wu/proj/Cyclegan_sn/data/same_density.pickle', "rb") as pic:
        #     pic_sub = pickle.load(pic, encoding='latin1')
        # for idx, image in pic_sub.items():
        #     image = N.array(image)
        #     for i in range(10):
        #         save_png(cub_img(image[i,:].reshape((40,40,40)))['im'], data+kind+'/'+idx+'_{}_real_den.png'.format(i))

        # with open('/shared/home/v_xindi_wu/proj/img2img_new/data/simulated1200.pickle', "rb") as pic:
        #     pic_sub = pickle.load(pic, encoding='latin1')
        fake_sub = N.load(os.path.join(root+kind, "best_acc_fake_subtomogram_test.npy"))
        label = N.load(os.path.join(root+kind, "density_map_label_test.npy"))
        pic_sub = {name:[] for idx,name in id2label.items()}
        for i in range(fake_sub.shape[0]):
            pic_sub[id2label[label[i]]].append(fake_sub[i,:].reshape(img_size,img_size,img_size))

        for idx, image in pic_sub.items():
            image = N.array(image)
            print(image.shape)
            if not os.path.exists(data+kind+'/'+idx):
                os.makedirs(data+kind+'/'+idx)
            for i in range(30):
                save_png(cub_img(image[i,:].reshape((img_size,img_size,img_size)))['im'], data+kind+'/'+idx+'/'+idx+'_{}_sub.png'.format(i))

if __name__ == '__main__':
    # main()
    uncertainty_est(40)
