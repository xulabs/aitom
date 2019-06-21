import numpy as np
cimport numpy as np
cimport cython

from libcpp.string cimport string

cdef extern from "wrap_core.hpp":
    cdef void wrap_write_mrc(double *, unsigned int, unsigned int, unsigned int, string) except +
    cdef void *wrap_read_mrc(string, double **, unsigned int *, unsigned int *, unsigned int *) except +
    cdef void *wrap_combined_search(unsigned int, unsigned int, unsigned int, double *, double *, double *, double *, unsigned int, unsigned int *, double **) except +
    cdef void *wrap_rot_search_cor(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *v1_data, double *v2_data, unsigned int n_radii, double *radii_data, unsigned int L, unsigned int *n_cor_r, unsigned int *n_cor_c, unsigned int *n_cor_s, double **cor) except +
    cdef void *wrap_local_max_angles(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *cor_data, unsigned int peak_spacing, unsigned int *n_res, double **res_data) except +
    cdef void wrap_rotate_vol_pad_mean(unsigned int, unsigned int, unsigned int, double *, double *, double *, double *) except +
    cdef void wrap_rotate_vol_pad_zero(unsigned int, unsigned int, unsigned int, double *, double *, double *, double *) except +
    cdef void wrap_rotate_mask(unsigned int, unsigned int, unsigned int, double *, double *, double *) except +
    cdef void wrap_del_cube(void *c) except +
    cdef void wrap_del_mat(void *c) except +
    cdef void wrap_ac_distance_transform_3d(const unsigned int n_r, const unsigned int n_c, const unsigned int n_s, const char *lbl_v, double *dist_v) except +
    cdef void wrap_BinaryBoundaryDetection(char *pI, int width, int height, int depth, int type, char *pOut) except + 
    cdef void wrap_ac_div_AOS_3D_dll(const unsigned int* dims, double *g_v, double *phi_v, double *phi_n_v, const double delta_t) except +
    cdef void wrap_watershed_segmentation(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *vol__data, int *lbl__data, const unsigned int max_overall_voxel_num, const unsigned int max_segment_voxel_num, const int queue_label, const int conflict_lbl, int*vol_seg_lbl__data, unsigned int *vol_num__data, unsigned int *overall_num__data) except +
    cdef int wrap_segment_boundary(unsigned int n_r, unsigned int n_c, unsigned int n_s, int *lbl__data, int *bdry__data) except +
    cdef int wrap_connected_regions(unsigned int n_r, unsigned int n_c, unsigned int n_s, int *msk__data, int *lbl__data) except +


@cython.boundscheck(False)
@cython.wraparound(False)
def write_mrc(np.ndarray[np.double_t, ndim=3] vol, str filename):
    """
    TODO: documentation
    """

    cdef double *vol_data
    cdef unsigned int n_r, n_c, n_s

    if not vol.flags.f_contiguous:
        vol = vol.copy(order='F')

    if vol.dtype != np.float:
        vol = vol.astype(np.float)

    vol_data = <double *>vol.data
    n_r = vol.shape[0]
    n_c = vol.shape[1]
    n_s = vol.shape[2]

    wrap_write_mrc(vol_data, n_r, n_c, n_s, filename)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
def read_mrc(str filename):
    """
    TODO: documentation
    """

    cdef double *v_data
    cdef unsigned int n_r, n_c, n_s
    cdef np.ndarray[np.double_t, ndim=3] vol
    cdef void *cube_ptr

    cube_ptr = wrap_read_mrc(filename, &v_data, &n_r, &n_c, &n_s)

    vol = np.empty( (n_r, n_c, n_s), dtype=np.double, order='F')

    cdef double *np_data = <double*> vol.data

    cdef size_t i
    for i in range(n_r*n_c*n_s):
        np_data[i] = v_data[i]

    wrap_del_cube(cube_ptr)

    return vol


@cython.boundscheck(False)
@cython.wraparound(False)
def combined_search(np.ndarray[np.double_t, ndim=3] vol1, np.ndarray[np.double_t, ndim=3] mask1, np.ndarray[np.double_t, ndim=3] vol2, np.ndarray[np.double_t, ndim=3] mask2, unsigned int L):
    """
    TODO: documentation
    """

    if vol1.max() == vol1.min():    raise RuntimeError('vol1.max() == vol1.min()')          # in such case, the alignment will stuck
    if vol2.max() == vol2.min():    raise RuntimeError('vol2.max() == vol2.min()')          # in such case, the alignment will stuck

    cdef double *v1_data
    cdef double *m1_data
    cdef double *v2_data
    cdef double *m2_data

    cdef double *res_data
    cdef unsigned int n_res

    cdef np.ndarray[np.double_t, ndim=2] res

    cdef void   *mat_ptr

    cdef unsigned int n_r, n_c, n_s

    if not vol1.flags.f_contiguous:
        vol1 = vol1.copy(order='F')
    if not mask1.flags.f_contiguous:
        mask1 = mask1.copy(order='F')
    if not vol2.flags.f_contiguous:
        vol2 = vol2.copy(order='F')
    if not mask2.flags.f_contiguous:
        mask2 = mask2.copy(order='F')

    v1_data = <double *> vol1.data
    m1_data = <double *>mask1.data
    v2_data = <double *> vol2.data
    m2_data = <double *>mask2.data

    n_r = vol1.shape[0]
    n_c = vol1.shape[1]
    n_s = vol1.shape[2]

    mat_ptr = wrap_combined_search(n_r, n_c, n_s, v1_data, m1_data, v2_data, m2_data, L, &n_res, &res_data)


    res = np.empty( (n_res, 7), dtype=np.double, order='F')

    cdef double *np_data = <double*> res.data

    cdef size_t i
    for i in range(n_res*7):
        np_data[i] = res_data[i]

    wrap_del_mat(mat_ptr)

    R = []

    for i in range(n_res):
        R.append((res[i,0], np.array(res[i,1:4]), np.array(res[i,4:])))
    return R





@cython.boundscheck(False)
@cython.wraparound(False)
def rot_search_cor(np.ndarray[np.double_t, ndim=3] v1, np.ndarray[np.double_t, ndim=3] v2, np.ndarray[np.double_t, ndim=1] radii, unsigned int L=36):

    if v1.max() == v1.min():    raise RuntimeError('v1.max() == v1.min()')          # in such case, the calcualtion may stuck
    if v2.max() == v2.min():    raise RuntimeError('v2.max() == v2.min()')          # in such case, the calcualtion may stuck


    if not v1.flags.f_contiguous:     v1 = v1.copy(order='F')
    if not v2.flags.f_contiguous:     v2 = v2.copy(order='F')

    cdef double *v1_data
    cdef double *v2_data
    v1_data = <double *> v1.data
    v2_data = <double *> v2.data



    cdef unsigned int n_r, n_c, n_s

    n_r = v1.shape[0]
    n_c = v1.shape[1]
    n_s = v1.shape[2]

    cdef unsigned int n_radii
    n_radii = len(radii)

    if not radii.flags.f_contiguous:        radii = radii.copy(order='F')
    cdef double *radii_data
    radii_data = <double *> radii.data


    cdef unsigned int n_cor_r, n_cor_c, n_cor_s
    cdef double *cor_data
    cdef void *cor_ptr

    cor_ptr = wrap_rot_search_cor(n_r, n_c, n_s, v1_data, v2_data, n_radii, radii_data, L, &n_cor_r, &n_cor_c, &n_cor_s, &cor_data)

    cdef np.ndarray[np.double_t, ndim=3] cor
    cor = np.empty( (n_cor_r, n_cor_c, n_cor_s), dtype=np.double, order='F')

    cdef double *cor_data_np = <double*> cor.data

    cdef size_t i
    for i in range(n_cor_r * n_cor_c * n_cor_s):
        cor_data_np[i] = cor_data[i]

    wrap_del_cube(cor_ptr)

    return cor




@cython.boundscheck(False)
@cython.wraparound(False)
def local_max_angles(np.ndarray[np.double_t, ndim=3] cor, unsigned int peak_spacing=8):

    print 'local_max_angles()'

    if not cor.flags.f_contiguous:     cor = cor.copy(order='F')

    cdef double *cor_data
    cor_data = <double *> cor.data

    cdef unsigned int n_r, n_c, n_s
    n_r = cor.shape[0]
    n_c = cor.shape[1]
    n_s = cor.shape[2]


    cdef unsigned int n_res
    cdef double *res_data

    res_ptr = wrap_local_max_angles(n_r, n_c, n_s, cor_data, peak_spacing, &n_res, &res_data)

    cdef np.ndarray[np.double_t, ndim=2] res
    res = np.empty( (n_res, 4), dtype=np.double, order='F')

    cdef double *res_data_np = <double*> res.data

    cdef size_t i
    for i in range(n_res*4):
        res_data_np[i] = res_data[i]

    wrap_del_mat(res_ptr)

    
    cors = res[:,0]
    angs = res[:,1:]

    return (cors, angs)


@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_vol_pad_mean(np.ndarray[np.double_t, ndim=3] vol, np.ndarray[np.double_t, ndim=1] ea, np.ndarray[np.double_t, ndim=1] dx):
    """
    TODO: documentation
    """

    cdef double *vol_data
    cdef double *ea_data
    cdef double *dx_data
    cdef double *res_data

    cdef np.ndarray[np.double_t, ndim=3] res

    cdef unsigned int n_r, n_c, n_s

    if not vol.flags.f_contiguous:
        vol = vol.copy(order='F')

    n_r = vol.shape[0]
    n_c = vol.shape[1]
    n_s = vol.shape[2]

    res = np.empty((n_r, n_c, n_s), dtype=np.double, order='F')

    vol_data = <double *>vol.data
    ea_data  = <double *> ea.data
    dx_data  = <double *> dx.data
    res_data = <double *>res.data

    wrap_rotate_vol_pad_mean(n_r, n_c, n_s, vol_data, ea_data, dx_data, res_data);
    return res



@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_vol_pad_zero(np.ndarray[np.double_t, ndim=3] vol, np.ndarray[np.double_t, ndim=1] ea, np.ndarray[np.double_t, ndim=1] dx):
    """
    TODO: documentation
    """

    cdef double *vol_data
    cdef double *ea_data
    cdef double *dx_data
    cdef double *res_data

    cdef np.ndarray[np.double_t, ndim=3] res

    cdef unsigned int n_r, n_c, n_s

    if not vol.flags.f_contiguous:
        vol = vol.copy(order='F')

    n_r = vol.shape[0]
    n_c = vol.shape[1]
    n_s = vol.shape[2]

    res = np.empty((n_r, n_c, n_s), dtype=np.double, order='F')

    vol_data = <double *>vol.data
    ea_data  = <double *> ea.data
    dx_data  = <double *> dx.data
    res_data = <double *>res.data

    wrap_rotate_vol_pad_zero(n_r, n_c, n_s, vol_data, ea_data, dx_data, res_data);
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_mask(np.ndarray[np.double_t, ndim=3] mask, np.ndarray[np.double_t, ndim=1] ea):
    """
    TODO: documentation
    """

    cdef double *mask_data
    cdef double *ea_data
    cdef double *res_data

    cdef np.ndarray[np.double_t, ndim=3] res

    cdef unsigned int n_r, n_c, n_s

    if not mask.flags.f_contiguous:
        mask = mask.copy(order='F')

    n_r = mask.shape[0]
    n_c = mask.shape[1]
    n_s = mask.shape[2]

    res = np.empty((n_r, n_c, n_s), dtype=np.double, order='F')

    mask_data = <double *>mask.data
    ea_data  = <double *> ea.data
    res_data = <double *>res.data

    wrap_rotate_mask(n_r, n_c, n_s, mask_data, ea_data, res_data);
    return res



@cython.boundscheck(False)
@cython.wraparound(False)
def ac_distance_transform_3d(np.ndarray[np.uint8_t, ndim=3] lbl):


    if not lbl.flags.f_contiguous:  lbl = lbl.copy(order='F')

    cdef unsigned int n_r = lbl.shape[0]
    cdef unsigned int n_c = lbl.shape[1]
    cdef unsigned int n_s = lbl.shape[2]

    cdef char *lbl_data = <char *>lbl.data


    cdef np.ndarray[np.double_t, ndim=3] dist = np.zeros((n_r, n_c, n_s), dtype=np.double, order='F')
    cdef double *dist_data = <double *>dist.data

    wrap_ac_distance_transform_3d(n_r, n_c, n_s, lbl_data, dist_data)

    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
def zy_binary_boundary_detection(np.ndarray[np.uint8_t, ndim=3] lbl, int type_t=1):


    if not lbl.flags.f_contiguous:  lbl = lbl.copy(order='F')

    cdef unsigned int n_r = lbl.shape[0]
    cdef unsigned int n_c = lbl.shape[1]
    cdef unsigned int n_s = lbl.shape[2]

    cdef char *lbl_data = <char *>lbl.data

    cdef np.ndarray[np.uint8_t, ndim=3] lbl_out = np.zeros((n_r, n_c, n_s), dtype=np.uint8, order='F')
    cdef char *lbl_out_data = <char *>lbl_out.data

   
    wrap_BinaryBoundaryDetection(lbl_data, n_r, n_c, n_s, type_t, lbl_out_data)

    return lbl_out


@cython.boundscheck(False)
@cython.wraparound(False)
def ac_div_AOS_3D(np.ndarray[np.double_t, ndim=3] phi, np.ndarray[np.double_t, ndim=3] g, double delta_t):

    if not phi.flags.f_contiguous:        phi = np.array(phi, order='F', dtype=np.float)
    cdef double *phi_data = <double *> phi.data

    if not g.flags.f_contiguous:        g = np.array(g, order='F', dtype=np.float)
    cdef double *g_data = <double *> g.data


    cdef unsigned int dims[3]
    dims[0] = phi.shape[0]
    dims[1] = phi.shape[1]
    dims[2] = phi.shape[2]

    cdef np.ndarray[np.double_t, ndim=3] phi_n = np.empty((phi.shape[0], phi.shape[1], phi.shape[2]), dtype=np.float, order='F')
    cdef double *phi_n_data = <double *> phi_n.data
    
    wrap_ac_div_AOS_3D_dll(dims, g_data, phi_data, phi_n_data, delta_t)

    return phi_n


@cython.boundscheck(False)
@cython.wraparound(False)
def watershed_segmentation(np.ndarray[np.double_t, ndim=3] vol, np.ndarray[np.int32_t, ndim=3] lbl, unsigned int max_overall_voxel_num, const unsigned int max_segment_voxel_num, const int queue_label, const int conflict_lbl):
    if not vol.flags['F_CONTIGUOUS']:      vol = np.array(vol, order='F', dtype=np.float)
    cdef double *vol__data = <double *> vol.data

    if not lbl.flags['F_CONTIGUOUS']:      lbl = np.array(lbl, order='F', dtype=np.int32)
    cdef int *lbl__data = <int *> lbl.data

    n_r = vol.shape[0]
    n_c = vol.shape[1]
    n_s = vol.shape[2]


    cdef np.ndarray[np.int32_t, ndim=3] vol_seg_lbl = np.empty([n_r, n_c, n_s], order='F', dtype=np.int32)
    cdef int *vol_seg_lbl__data = <int *> vol_seg_lbl.data

    cdef np.ndarray[np.uint32_t, ndim=3] vol_num = np.empty([n_r, n_c, n_s], order='F', dtype=np.uint32)
    cdef unsigned int *vol_num__data = <unsigned int *> vol_num.data

    cdef np.ndarray[np.uint32_t, ndim=3] overall_num = np.zeros([n_r, n_c, n_s], order='F', dtype=np.uint32)
    cdef unsigned int *overall_num__data = <unsigned int *> overall_num.data


    wrap_watershed_segmentation(n_r, n_c, n_s, vol__data, lbl__data, max_overall_voxel_num, max_segment_voxel_num, queue_label, conflict_lbl, vol_seg_lbl__data, vol_num__data, overall_num__data)

    return {'vol_seg_lbl':vol_seg_lbl, 'vol_num':vol_num, 'overall_num':overall_num}


@cython.boundscheck(False)
@cython.wraparound(False)
def segment_boundary(np.ndarray[np.int32_t, ndim=3] lbl):
    if not lbl.flags['F_CONTIGUOUS']:      lbl = np.array(lbl, order='F', dtype=np.int32)
    cdef int *lbl__data = <int *> lbl.data

    n_r = lbl.shape[0]
    n_c = lbl.shape[1]
    n_s = lbl.shape[2]

    cdef np.ndarray[np.int32_t, ndim=3] bdry = np.empty([n_r, n_c, n_s], order='F', dtype=np.int32)
    cdef int *bdry__data = <int *> bdry.data

    count = wrap_segment_boundary(n_r, n_c, n_s, lbl__data, bdry__data)


    return {'bdry':bdry, 'count':count}


'''
interface for calculating connected regions.
Note: there is a function for processing 2D image https://github.com/scikit-image/scikit-image/blob/master/skimage/morphology/ccomp.pyx
'''
@cython.boundscheck(False)
@cython.wraparound(False)
def connected_regions(np.ndarray[np.int32_t, ndim=3] msk):

    if not msk.flags['F_CONTIGUOUS']:      msk = np.array(msk, order='F', dtype=np.int32)
    cdef int *msk__data = <int *> msk.data

    n_r = msk.shape[0]
    n_c = msk.shape[1]
    n_s = msk.shape[2]

    cdef np.ndarray[np.int32_t, ndim=3] lbl = np.zeros([n_r, n_c, n_s], order='F', dtype=np.int32)
    cdef int *lbl__data = <int *> lbl.data

    max_lbl = wrap_connected_regions(n_r, n_c, n_s, msk__data, lbl__data)

    return {'max_lbl':max_lbl, 'lbl':lbl}


'''
count the number of labels inside a volume
we assume label 0 correspond to empty region
'''
@cython.boundscheck(False)
@cython.wraparound(False)
def vol_label_count(np.ndarray[np.int32_t, ndim=3] lbl):

    if not lbl.flags['F_CONTIGUOUS']:      lbl = np.array(lbl, order='F', dtype=np.int32)

    c = [0] * (lbl.max()+1)
    for i0 in xrange(lbl.shape[0]):
        for i1 in xrange(lbl.shape[1]):
            for i2 in xrange(lbl.shape[2]):
                if lbl[i0,i1,i2] > 0:                c[lbl[i0,i1,i2]] += 1

    return c



