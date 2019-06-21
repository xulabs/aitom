
#ifndef WATERSHED_SEGMENTATION_HPP__
#define WATERSHED_SEGMENTATION_HPP__


int watershed_segmentation(const arma::cube &vol, const arma::icube &lbl, const unsigned int max_overall_voxel_num, const unsigned int max_segment_voxel_num, const int queue_label, const int conflict_lbl, arma::icube &vol_seg_lbl, arma::ucube &vol_num, arma::ucube &overall_num);

int segment_boundary(const arma::icube &lbl, arma::icube &bdry);

int connected_regions(const arma::icube &msk, arma::icube &lbl);

#endif

