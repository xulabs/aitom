'''
Authors of the code: Xiangrui Zeng, Min Xu
License: ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

Reference:
Zeng X, Leung M, Zeev-Ben-Mordehai T, Xu M. A convolutional autoencoder approach for mining features in cellular electron cryo-tomograms and weakly supervised coarse segmentation. Journal of Structural Biology (2017) doi:10.1016/j.jsb.2017.12.015

Please cite the above paper when this code is used or adapted for your research.

'''

#import EMAN2 as E
import mrcfile
import scipy.ndimage as SN
import scipy.ndimage.filters as SNF
import gc as GC
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from scipy.spatial.distance import cdist
import numpy as N
import scipy.ndimage.interpolation as SNI
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import os, sys, json, shutil, time, struct, array, pickle
import copy
import math

def pickle_load(path):
    with open(path,'rb') as f:
        o = pickle.load(f,encoding='iso-8859-1')
    return o

def pickle_dump(o, path, protocol=-1):
    with open(path, 'wb') as f:
        pickle.dump(o, f, protocol=protocol)



# def em2numpy(ve):
#     assert ve.get_ndim() == 3           # only for 3D array
#     return E.EMNumPy.em2numpy(ve).transpose([2,1,0]).copy()

'''
read mrc 3D image file into a numpy 3D array
'''
# def read_mrc_numpy_vol_(path):
#     ve = E.EMData(str(path))
#     v = em2numpy(ve)
#     v = v.astype(N.float32)
#     del ve      # to prevent memory leak
#     return v

def peaks_to_subvolumes(v, peaks, subvol_size=32):
    """Convert particle picking peaks to subvolumes. Can be later used as autoencoder input. 

    Returns:
        d is the small subvolumes, a dictionary consists 'v_siz' and 'vs'.
            d['v_siz'] is an numpy.ndarray specifying the shape of the small subvolume. For example, d['v_siz'] = array([32,32,32]).
            d['vs'] is a dictionary with keys of uuids specifying each small subvolume.
            d['vs'][an example uuid] is a dictionary consists 'center', 'id', and 'v'.
            d['vs'][an example uuid]['center'] is the center of the small subvolume in the tomogram. For example, d['vs'][an example uuid]['center'] = [110,407,200].
            d['vs'][an example uuid]['id'] is the specific uuid.
            d['vs'][an example uuid]['v'] are voxel values of the small subvolume, which is an numpy.ndarray of shape d['v_siz'],
                notice that if subvol has part outside vol, then ['v'] = None
    """
    d = {}
    d['v_siz'] = subvol_size
    d['vs'] = {}
    for p in peaks:
        uuid = p['uuid']
        d['vs'][uuid] = {}
        d['vs'][uuid]['center'] = p['x']
        d['vs'][uuid]['id'] = uuid
        d['vs'][uuid]['v'] = cut_from_whole_map(v, p['x'], subvol_size)
    return d
    
    
def read_mrc_numpy_vol(path):
    with mrcfile.open(path) as mrc:
        v = mrc.data
        v = v.astype(N.float32).transpose([2,1,0])
    return v


# 3D gaussian filtering of a volume (v)
# smoothing using scipy.ndimage.gaussian_filter
def smooth(v, sigma):
    assert  sigma > 0
    return SN.gaussian_filter(input=v, sigma=sigma)


# paste to a map given start and end coordinates
def paste_to_whole_map__se(whole_map, vol, se):
    whole_map[se[0,0]:se[0,1], se[1,0]:se[1,1], se[2,0]:se[2,1]] = vol


def difference_of_gauss_function(size, sigma1, sigma2):

    grid = grid_displacement_to_center(size)
    dist_sq = grid_distance_sq_to_center(grid)

    del grid

    dog = (1 / ( (2 * N.pi)**(3.0/2.0)  * (sigma1**3)) ) * N.exp( - (dist_sq)  / (2.0 * (sigma1**2)))               # gauss function
    dog -= (1 / ( (2 * N.pi)**(3.0/2.0)  * (sigma2**3)) ) * N.exp( - (dist_sq)  / (2.0 * (sigma2**2)))

    return dog


def grid_displacement_to_center(size, mid_co=None):

    size = N.array(size, dtype=N.float)
    assert size.ndim == 1

    if mid_co is None:            mid_co = (N.array(size) - 1) / 2          # IMPORTANT: following python convension, in index starts from 0 to size-1!!! So (siz-1)/2 is real symmetry center of the volume
 
    if size.size == 3:
        # construct a gauss function whose center is at center of volume
        grid = N.mgrid[0:size[0], 0:size[1], 0:size[2]]

        for dim in range(3):
            grid[dim, :, :, :] -= mid_co[dim]

    elif size.size == 2:
        # construct a gauss function whose center is at center of volume
        grid = N.mgrid[0:size[0], 0:size[1]]

        for dim in range(2):
            grid[dim, :, :] -= mid_co[dim]

    else:
        assert False

    return grid

def grid_distance_sq_to_center(grid):
    dist_sq = N.zeros(grid.shape[1:])
    if grid.ndim == 4:
        for dim in range(3):
            dist_sq += N.squeeze(grid[dim, :, :, :]) ** 2
    elif grid.ndim == 3:
        for dim in range(2):
            dist_sq += N.squeeze(grid[dim, :, :]) ** 2
    else:
        assert False

    return dist_sq


def put_mrc(mrc, path):
    put_mrc_data(mrc,path)

def put_mrc_data(mrc,path):
    write_data(data=mrc,path=path)

def write_data(data, path):
    assert data.ndim == 3  # only for 3D array

    data = data.astype(N.float32)
    data = data.transpose([2,1,0])        # this is according to tomominer.image.vol.eman2_util.numpy2em
    with mrcfile.new(path) as m:
        m.set_data(data)




def cut_from_whole_map(whole_map, c, siz):

    se = subvolume_center_start_end(c, map_siz=whole_map.shape, subvol_siz=siz)
    return          cut_from_whole_map__se(whole_map, se)


# cut a map given start and end coordinates
def cut_from_whole_map__se(whole_map, se):
    if se is None:       return None
    return          whole_map[se[0,0]:se[0,1], se[1,0]:se[1,1], se[2,0]:se[2,1]]



# given a center c, get the relative start and end position of a subvolume with size subvol_siz
def subvolume_center_start_end(c, map_siz, subvol_siz):
    map_siz = N.array(map_siz)
    subvol_siz = N.array(subvol_siz)

    siz_h = N.ceil( subvol_siz / 2.0 )

    start = c - siz_h;      start.astype(int)
    end = start + subvol_siz;      end.astype(int)

    if any(start < 0):  return None
    if any(end >= map_siz):    return None

    se = N.zeros( (3,2), dtype=N.int )
    se[:,0] = start
    se[:,1] = end

    return se

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
        print ('save_png()')
        print ('unique values', sorted(set(m.flatten())))

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

def center_mass(v):
    assert  N.all(v >= 0)
    
    m = v.sum()
    assert  m > 0

    v = v/m

    s = v.shape
    g = N.mgrid[0:s[0], 0:s[1], 0:s[2]]

    c = [None] * v.ndim
    for dim_i in range(v.ndim):
        c[dim_i] = (g[dim_i] * v).sum()


    return N.array(c)



'''
perform pose normalization through PCA, assume that the values in V is non-negative
'''
def pca(v, c, do_flip=False):
    assert      N.all(v >= 0)

    re = {'c':c}

    s = v.shape

    g = N.mgrid[0:s[0], 0:s[1], 0:s[2]]
    g = N.array(g, dtype=N.float)

    for i in range(len(g)):     g[i] -= c[i]

    gv = []
    
    gv = [g[0].flatten(), g[1].flatten(), g[2].flatten()]
    
    vv = v.flatten()
    gvw = [_*vv for _ in gv]

    wsm = N.dot(  N.array(gv),    N.array(gvw).T   )      # weighted sum matrix
    re['wsm'] = wsm


    # perform eigen decomposition, and order eigen values and eigen vectors according to decreasing order of magnitude of eignenvalues
    (eig_w, eig_v) = N.linalg.eig(wsm)

    i = N.argsort(-N.abs(eig_w))        # the resulting projection vectors in the rotation matrix will be ordered by the magnitude of eigenvalues
    eig_w = eig_w[i]
    eig_v = eig_v[:, i]


    re['w'] = eig_w

    if do_flip:
        pass
        #re['v'] = flip_sign(v=v, c=c, r=eig_v)
    else:
        re['v'] = eig_v         # this is the rotation matrix. rotate vol by rm gives pose normalized vol

    return re


def concat(s0=None, s1=None):

    if s0 is None:      return copy.deepcopy(s1)
    if s1 is None:      return copy.deepcopy(s0)

    n0 = s0['vertices'].shape[0]
    n1 = s1['vertices'].shape[1]

    s = {}
    s['vertices'] = N.vstack(   (s0['vertices'], s1['vertices'])    )
    s['faces'] = N.vstack(  (s0['faces'], s1['faces']+n0)   )

    # add a class label field so that in future the surfaces can be decomposed
    if 'class' not in s0:   s0['class'] = N.ones(n0)
    if 'class' not in s1:   s1['class'] = N.ones(n1)

    s['class'] = N.hstack(      (s0['class'], s1['class'] + s0['class'].max() + 1)      )

    return s


def concat_list(ss):
    s = None
    for st in ss:       s = concat(s, st)
    
    return s


def rotate_retrieve(v, tom, center = None, angle = None, rm=None, c1=None, c2=None, loc_r=None, siz2=None, default_val=float('NaN')):
    if center is None:
        center = [0,0,0]    
    

    if angle is not None:
        assert      rm is None
        angle = N.array(angle, dtype=N.float).flatten()
        rm = rotation_matrix_zyz(angle)
      
    siz = N.ceil((v.shape[0]+ 2*max(N.abs(loc_r)))*N.abs(rm.dot([1,1,1])).max()) #bounding box size
    if all((center - siz) > 0) and all((center + siz) < tom.shape): #test if the tomogram enclose the bounding box
        vb = cut_from_whole_map(tom, center, siz)
    
    else: 
        vb = N.full((int(siz),int(siz),int(siz)),default_val) #set non-enclosed regions as default value
        siz_h = N.ceil( siz / 2.0 )

        start = center - siz_h;      start.astype(int)
        end = start + siz;      end.astype(int)
        
        start_vb = start.copy()
        start[N.where(start < 0)] = 0
        end[N.where(end > tom.shape)] = N.array(tom.shape)[end > tom.shape]

        se = N.zeros( (3,2), dtype=N.int )
        se[:,0] = start
        se[:,1] = end
        
        se_vb = se.copy()

        se_vb[:,0] = se_vb[:,0] - start_vb
        se_vb[:,1] = se_vb[:,1] - start_vb
         
        vb[se_vb[0,0]:se_vb[0,1], se_vb[1,0]:se_vb[1,1], se_vb[2,0]:se_vb[2,1]] = tom[se[0,0]:se[0,1], se[1,0]:se[1,1], se[2,0]:se[2,1]]

    if rm is None:      rm = N.eye(vb.ndim)

    siz1 = N.array( vb.shape, dtype=N.float )
    if c1 is None:
        c1 = (siz1-1) / 2.0                  # IMPORTANT: following python convension, in index starts from 0 to size-1!!! So (siz-1)/2 is real symmetry center of the volume
    else:
        c1 = c1.flatten()
    assert  c1.shape == (3,)

    if siz2 is None:    siz2 = siz1
    siz2 = N.array(siz2, dtype=N.float)

    if c2 is None:      
        c2 = (siz2-1) / 2.0               # IMPORTANT: following python convension, in index starts from 0 to size-1!!! So (siz-1)/2 is real symmetry center of the volume
    else:
        c2 = c2.flatten()
    assert c2.shape == (3,)

    if loc_r is not None:
        loc_r = N.array(loc_r, dtype=N.float).flatten()
        assert  loc_r.shape == (3,)
        c2 += loc_r


    
    c = -rm.dot(c2) + c1

    #rm_ext = N.hstack( (rm, c) )

    vbr = SNI.affine_transform(input=vb, matrix=rm, offset=c, output_shape=siz2.astype(N.int), cval=default_val)          # note: output_shape need to be integers, otherwise N.zeros will raise     
    
    c = N.array([(vbr.shape[0])/2]*3, dtype=int)
    siz = v.shape[0]
    vr = cut_from_whole_map(vbr, c, siz) #crop the center image of original subtomogram size from the rotated bounding box
    
    return vr

def rotation_matrix_zyz(ang):
    phi = ang[0];       theta = ang[1];     psi_t = ang[2];
    
    a1 = rotation_matrix_axis(2, psi_t)       # first rotate about z axis for angle psi_t
    a2 = rotation_matrix_axis(1, theta)
    a3 = rotation_matrix_axis(2, phi)
    
    rm = a3.dot(a2).dot(a1)      # for matrix left multiplication
    
    rm = rm.transpose()       # note: transform because tformarray use right matrix multiplication

    return rm

def rotation_matrix_axis(dim, theta):
    # following are left handed system (clockwise rotation)
    # IMPORTANT: different to MATLAB version, this dim starts from 0, instead of 1
    if dim == 0:        # x-axis
        rm = N.array(  [[1.0, 0.0, 0.0], [0.0, math.cos(theta), -math.sin(theta)], [0.0, math.sin(theta), math.cos(theta)]]  )
    elif dim == 1:    # y-axis
        rm = N.array(  [[math.cos(theta), 0.0, math.sin(theta)], [0.0, 1.0, 0.0], [-math.sin(theta), 0.0, math.cos(theta)]]  )
    elif dim == 2:        # z-axis
        rm = N.array(  [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]]  )
    else:
        #raise
        pass
    
    return rm
