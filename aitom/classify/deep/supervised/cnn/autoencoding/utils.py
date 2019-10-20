import pickle
import numpy as N 

def pickle_load(path):
    with open(path) as f:     o = pickle.load(f)
    return o

def visualize(img, path):
	t = cub_img(img)['im']
	save_png(t, path)

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

def save_png(m, name, normalize=True, verbose=False):

    if verbose:
        print( 'save_png()')
        print( 'unique values', sorted(set(m.flatten())))

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

    import png          # in pypng package
    png.from_array(m, 'L').save(name)