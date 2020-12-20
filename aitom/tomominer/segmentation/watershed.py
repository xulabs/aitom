"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import numpy as np
import aitom.tomominer.core.core as core


def segmentation_cpp(vol_map, vol_lbl, max_overall_voxel_num=None, max_segment_voxel_num=None, queue_label=(-1),
                     conflict_lbl=(-2)):
    vol_map = np.array(vol_map, dtype=np.double, order='F')
    vol_lbl = np.array(vol_lbl, dtype=np.int32, order='F')
    if max_overall_voxel_num is None:
        max_overall_voxel_num = (vol_lbl.size + 1)
    if max_segment_voxel_num is None:
        max_segment_voxel_num = (vol_lbl.size + 1)
    return core.watershed_segmentation(vol_map, vol_lbl, max_overall_voxel_num, max_segment_voxel_num, queue_label,
                                       conflict_lbl)


def segment(*args, **kwargs):
    return segmentation_cpp(*args, **kwargs)
