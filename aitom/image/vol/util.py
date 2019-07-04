
# convert a 3D cube to a 2D image of slices
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

# display an image
def dsp_img(v, new_figure=True):
    import tomominer.image.util as TIU
    TIU.dsp_img(v, new_figure=new_figure)


def dsp_cub(v, view_dir=2, new_figure=True):

    dsp_img(cub_img(v=v, view_dir=view_dir)['im'])


'''
make a toy structure, and highlight the positive part of Y axis, which is supposed to be tiltling about
also highlight the positive part of x-axis
'''
def highlight_xy_axis(v, dim_siz=64, model_id=0, copy=True):
    if copy:    v = v.copy()

    c = N.array(v.shape) / 2
    c2 = c / 2

    m = N.abs(v).max()
    v[c[0]:,c[1],c[2]] = m      # highlight positive part of x-axis
    v[c[0]+c2[0], c[1]-c2[1]:c[1]+c2[1], c[2]] = m      # highlight positive part of x-axis, by adding a short segment in the middle of x axis

    v[c[0],c[1]:,c[2]] = m      # highlight positive part of y-axis

    return v




