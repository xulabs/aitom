# split_membranes.py
# Performs 2D slice-by-slice analysis of membrane shapes using PCA eigenvalue ratios and line fitting
# Classifies membrane regions into circular, straight lines, and other shapes (including curved lines)
# Usage:
#   python split_membranes.py TS_0008.mrc TS_0008_out_circular.mrc TS_0008_out_noncircular.mrc TS_0008_out_straight.mrc \
#          --label 1 --min_size 200 --roundness 0.5 --linearity 0.2 --straightness 40.0 --voxel_size 1.38

import argparse
import numpy as np
import mrcfile
from scipy import ndimage as ndi

def pca_eigenvalues_2d(coords_yx: np.ndarray, voxel_size_xy):
    if coords_yx.shape[0] < 3:
        return np.array([0.0, 0.0])
    # coords_yx is (N, y, x) from np.argwhere on 2D slice
    coords_xy = coords_yx[:, ::-1].astype(np.float64)  # (N, x, y)
    # scale to physical units (Å) so PCA is metric-correct
    sx, sy = voxel_size_xy
    coords_xy[:, 0] *= sx
    coords_xy[:, 1] *= sy
    X = coords_xy - coords_xy.mean(axis=0, keepdims=True)
    C = (X.T @ X) / (len(X) - 1)
    vals = np.linalg.eigvalsh(C)
    return np.sort(vals)[::-1] 

def is_roundish_2d(eigs, thresh=0.5):
    """
    Round criterion for 2D membrane regions:
    - isotropy: l2/l1 >= thresh (how circular vs elongated)
    """
    l1, l2 = eigs
    if l1 <= 0:
        return False
    r2 = l2/l1
    return r2 >= thresh

def is_linear_2d(eigs, thresh=0.3):
    """
    Linear/straight line criterion for 2D membrane regions:
    - strong anisotropy: l2/l1 <= thresh (very elongated)
    """
    l1, l2 = eigs
    if l1 <= 0:
        return False
    r2 = l2/l1
    return r2 <= thresh

def measure_line_straightness(coords_yx, voxel_size_xy):
    """
    Measure how straight a line is by fitting a line and measuring residuals
    Returns: straightness_score (higher = straighter)
    """
    if len(coords_yx) < 3:
        return 0.0
    
    # Convert to x,y coordinates and scale to physical units
    coords_xy = coords_yx[:, ::-1].astype(np.float64)  # (N, x, y)
    sx, sy = voxel_size_xy
    coords_xy[:, 0] *= sx
    coords_xy[:, 1] *= sy
    
    # Fit a line using PCA to get the main direction
    center = coords_xy.mean(axis=0)
    centered = coords_xy - center
    
    # Use SVD for more stable computation
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    if len(s) < 2:
        return 0.0
        
    direction = Vt[0]  # First principal component (main direction)
    
    # Project points onto the line
    projections = np.dot(centered, direction)
    projected_points = np.outer(projections, direction) + center
    
    # Calculate perpendicular distances to the fitted line
    residuals = np.linalg.norm(coords_xy - projected_points, axis=1)
    
    # Straightness metric: inverse of mean residual normalized by length
    mean_residual = np.mean(residuals)
    length = np.max(projections) - np.min(projections)
    
    if length <= 0 or mean_residual <= 0:
        return 0.0
    
    # Higher straightness = lower relative deviation
    straightness = length / mean_residual
    return straightness

def is_straight_line(eigs, coords_yx, voxel_size_xy, linearity_thresh=0.3, straightness_thresh=10.0):
    """
    Determine if a region is a straight line by combining:
    1. Elongation criterion (low eigenvalue ratio)
    2. Straightness criterion (low deviation from fitted line)
    """
    # Must be elongated first
    if not is_linear_2d(eigs, linearity_thresh):
        return False
    
    # Then check if it's straight (not curved)
    straightness = measure_line_straightness(coords_yx, voxel_size_xy)
    return straightness >= straightness_thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_mrc")
    ap.add_argument("out_circular_mrc")
    ap.add_argument("out_noncircular_mrc")
    ap.add_argument("out_linear_mrc")
    ap.add_argument("--label", type=int, default=1, help="voxel value used for membrane")
    ap.add_argument("--min_size", type=int, default=200, help="ignore components smaller than this (voxels)")
    ap.add_argument("--roundness", type=float, default=0.5, help="eigenvalue ratio threshold for circularity (0.6–0.85 sensible)")
    ap.add_argument("--linearity", type=float, default=0.3, help="eigenvalue ratio threshold for linearity (0.2–0.4 sensible)")
    ap.add_argument("--straightness", type=float, default=10.0, help="straightness threshold for distinguishing straight vs curved lines (5-20 sensible)")
    ap.add_argument("--voxel_size", type=float, default=1.0, help="voxel size in nm")
    ap.add_argument("--connectivity", type=int, default=2, choices=[1,2],
                    help="1=4N, 2=8N connectivity for 2D labeling")
    args = ap.parse_args()

    # --- Load segmentation
    with mrcfile.open(args.in_mrc, permissive=True) as mrc:
        vol = mrc.data.copy()
        header = mrc.header.copy()
    
    voxel_size_xy = (float(args.voxel_size), float(args.voxel_size))

    # --- Binary mask of the membrane label
    mask = (vol == args.label)
    
    # --- 2D connectivity structure
    structure_2d = np.ones((3,3), dtype=bool) if args.connectivity==2 else ndi.generate_binary_structure(2,1)
    
    out_circ = np.zeros_like(vol, dtype=vol.dtype)
    out_noncirc = np.zeros_like(vol, dtype=vol.dtype)
    out_linear = np.zeros_like(vol, dtype=vol.dtype)
    
    total_components = 0
    
    # --- Process each Z-slice independently
    print(f"Processing {vol.shape[0]} slices...")
    for z_idx in range(vol.shape[0]):
        slice_mask = mask[z_idx]
        
        if not slice_mask.any():
            continue
            
        # Apply morphological closing to connect nearby regions
        closed_slice = ndi.binary_closing(slice_mask, structure=ndi.generate_binary_structure(2,1), iterations=4)
        
        # Find connected components in this slice
        lab_slice, nlab_slice = ndi.label(closed_slice, structure=structure_2d)
        
        if nlab_slice == 0:
            continue
            
        # --- Iterate components in this slice
        for comp_id in range(1, nlab_slice+1):
            comp_mask = (lab_slice == comp_id)
            size = int(comp_mask.sum())
            if size < args.min_size:
                continue

            coords_yx = np.argwhere(comp_mask)  # (N, y, x)

            # PCA **in physical units** using yx coords
            eigs = pca_eigenvalues_2d(coords_yx, voxel_size_xy)

            # Classify based on eigenvalue ratios and straightness
            if is_roundish_2d(eigs, args.roundness):
                out_circ[z_idx][comp_mask] = args.label
                kind = "circular"
                straightness = 0.0  # Not applicable for circular
            elif is_straight_line(eigs, coords_yx, voxel_size_xy, args.linearity, args.straightness):
                out_linear[z_idx][comp_mask] = args.label
                kind = "straight"
                straightness = measure_line_straightness(coords_yx, voxel_size_xy)
            else:
                out_noncirc[z_idx][comp_mask] = args.label
                # Check if it was elongated but curved
                if is_linear_2d(eigs, args.linearity):
                    kind = "curved"
                    straightness = measure_line_straightness(coords_yx, voxel_size_xy)
                else:
                    kind = "non-circular"
                    straightness = 0.0  # Not applicable

            l1, l2 = eigs
            r2 = l2/l1 if l1 > 0 else 0
            total_components += 1
            print(f"Slice {z_idx:3d} | Comp {comp_id:3d} | voxels={size:6d} | eigs=({l1:.1f},{l2:.1f}) | r2={r2:.3f} | straight={straightness:.1f} -> {kind}")
    
    print(f"Processed {total_components} components across all slices")

    # --- Save MRCs (preserve voxel size & header basics)
    def save_mrc(path, data):
        with mrcfile.new(path, overwrite=True) as m:
            m.set_data(data.astype(vol.dtype))
            # keep the same voxel size (Å in MRC standard; mrcfile stores in Å)
            m.voxel_size = (voxel_size_xy[0], voxel_size_xy[1], args.voxel_size)

    save_mrc(args.out_circular_mrc, out_circ)
    save_mrc(args.out_noncircular_mrc, out_noncirc)
    save_mrc(args.out_linear_mrc, out_linear)
    print("Done.")

if __name__ == "__main__":
    main()
