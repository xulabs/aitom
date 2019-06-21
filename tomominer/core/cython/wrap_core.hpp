#ifndef TOMO_CYTHON_HPP
#define TOMO_CYTHON_HPP

#include <string>

void wrap_write_mrc(double *vol, unsigned int n_r, unsigned int n_c, unsigned int n_s, std::string filename);
void *wrap_read_mrc(std::string filename, double **vol, unsigned int *n_r, unsigned int *n_c, unsigned int *n_s);

void *wrap_combined_search(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *v1_data, double *m1_data, double *v2_data, double *m2_data, unsigned int L, unsigned int *n_res, double **res_data);
void *wrap_rot_search_cor(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *v1_data, double *v2_data, unsigned int n_radii, double *radii_data, unsigned int L, unsigned int *n_cor_r, unsigned int *n_cor_c, unsigned int *n_cor_s, double **cor);
void *wrap_local_max_angles(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *cor_data, unsigned int peak_spacing, unsigned int *n_res, double **res_data);

void wrap_rotate_vol_pad_mean(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *v_data, double *rm_data, double *dx_data, double *res_data);
void wrap_rotate_vol_pad_zero(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *v_data, double *rm_data, double *dx_data, double *res_data);
void wrap_rotate_mask(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *v_data, double *rm_data, double *res_data);
void wrap_del_cube(void *c);
void wrap_del_mat(void *v);

void wrap_ac_distance_transform_3d(const unsigned int n_r, const unsigned int n_c, const unsigned int n_s, const char *lbl_v, double *dist_v);
void wrap_BinaryBoundaryDetection(char *pI, int width, int height, int depth, int type, char *pOut); 
void wrap_ac_div_AOS_3D_dll(const unsigned int* dims, double *g_v, double *phi_v, double *phi_n_v, const double delta_t);


int wrap_watershed_segmentation(unsigned int n_r, unsigned int n_c, unsigned int n_s, double *vol__data, int *lbl__data, const unsigned int max_overall_voxel_num, const unsigned int max_segment_voxel_num, const int queue_label, const int conflict_lbl, int*vol_seg_lbl__data, unsigned int *vol_num__data, unsigned int *overall_num__data);
int wrap_segment_boundary(unsigned int n_r, unsigned int n_c, unsigned int n_s, int *lbl__data, int *bdry__data);
int wrap_connected_regions(unsigned int n_r, unsigned int n_c, unsigned int n_s, int *msk__data, int *lbl__data);


#endif // guard
