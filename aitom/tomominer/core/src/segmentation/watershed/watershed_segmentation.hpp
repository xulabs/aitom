
#ifndef WATERSHED_SEGMENTATION_HPP__
#define WATERSHED_SEGMENTATION_HPP__


int watershed_segmentation(const arma::cube &vol, const arma::s32_cube &lbl, const unsigned int max_overall_voxel_num, const unsigned int max_segment_voxel_num, const int queue_label, const int conflict_lbl, arma::s32_cube &vol_seg_lbl, arma::u32_cube &vol_num, arma::u32_cube &overall_num);

int segment_boundary(const arma::s32_cube &lbl, arma::s32_cube &bdry);

int connected_regions(const arma::s32_cube &msk, arma::s32_cube &lbl);

#endif

