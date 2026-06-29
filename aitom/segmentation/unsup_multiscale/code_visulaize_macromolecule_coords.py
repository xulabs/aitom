import os
import numpy as np
import mrcfile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─── USER SETTINGS FOR TS_0008 VISUALIZATION ──────────────────────────────────
# Update these paths to point to your local data and pipeline output directories.
# GROUND_TRUTH_PATH : .coords file with annotated ribosome positions
# UNSUP_PATH        : CSV produced by unified_tomogram_pipeline.py
# MRC_PATH          : raw MRC tomogram file
# OUTPUT_DIR        : directory where visualization PNGs will be saved
GROUND_TRUTH_PATH = 'input/ground_truth/TS_0008.coords'
UNSUP_PATH = 'output/TS_0008/macromolecule_coordinates/TS_0008_unsup_coords_membrane_filtered.csv'
MRC_PATH = 'input/raw_data/TS_0008.mrc'
OUTPUT_DIR = 'output/visualizations/TS_0008'

SIZE_NM = 30.0
VOXEL_SPACING = 1.34
Z_SLICE = 250
# ───────────────────────────────────────────────────────────────────────────────

def read_data(path):
    with mrcfile.open(path, mode='r', permissive=True) as mrc:
        vol = mrc.data.copy()
    assert vol.ndim == 3, "Expected a 3D tomogram"
    return vol

def read_ground_truth_coords(csv_path):
    coords = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    coords.append([x, y, z])
    return np.array(coords).round().astype(int)

def read_unsup_coords(csv_path):
    coords = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.replace(',', ' ').split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        coords.append([x, y, z])
                    except ValueError:
                        continue
    return np.array(coords).round().astype(int)

def draw_overlay(coords, vol, z_slice, size_nm, voxel_spacing, color, label, output_path):
    z0 = z_slice
    print(f"Visualizing {label} on z-slice: {z0}")

    size_v = int(size_nm / voxel_spacing)
    half = size_v // 2

    mask = np.abs(coords[:, 2] - z0) <= half
    xy = coords[mask, :2]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(vol[z0], cmap='gray')

    for x, y in xy:
        rect = patches.Rectangle(
            (x - half, y - half), size_v, size_v,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

    ax.axis('off')
    plt.title(f"{label} - TS_0008 slice {z0} ({len(xy)} particles)", fontsize=16, pad=20)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"{label} overlay saved to {output_path}")
    print(f"Found {len(xy)} particles on this slice (from {len(coords)} total)")

def draw_combined_overlay(gt_coords, unsup_coords, vol, z_slice, size_nm, voxel_spacing, output_path):
    z0 = z_slice
    size_v = int(size_nm / voxel_spacing)
    half = size_v // 2

    gt_mask = np.abs(gt_coords[:, 2] - z0) <= half
    unsup_mask = np.abs(unsup_coords[:, 2] - z0) <= half

    gt_xy = gt_coords[gt_mask, :2]
    unsup_xy = unsup_coords[unsup_mask, :2]

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(vol[z0], cmap='gray')

    for x, y in gt_xy:
        rect = patches.Rectangle(
            (x - half, y - half), size_v, size_v,
            linewidth=2, edgecolor='lime', facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)

    for x, y in unsup_xy:
        rect = patches.Rectangle(
            (x - half, y - half), size_v, size_v,
            linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)

    ax.axis('off')
    legend_elements = [
        patches.Patch(color='lime', label=f'Ground Truth ({len(gt_xy)})'),
        patches.Patch(color='yellow', label=f'Unsupervised ({len(unsup_xy)})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.title(f"Combined - TS_0008 slice {z0}", fontsize=18, pad=20)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined overlay saved to {output_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MRC_PATH):
        print(f"MRC file not found: {MRC_PATH}")
        return

    print("Loading tomogram...")
    vol = read_data(MRC_PATH)
    print(f"Tomogram shape: {vol.shape}")

    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Ground truth file not found: {GROUND_TRUTH_PATH}")
        return
    if not os.path.exists(UNSUP_PATH):
        print(f"Unsupervised file not found: {UNSUP_PATH}")
        return

    print("Loading ground truth coordinates...")
    gt_coords = read_ground_truth_coords(GROUND_TRUTH_PATH)
    print(f"Ground truth: {len(gt_coords)} particles")

    print("Loading unsupervised coordinates...")
    unsup_coords = read_unsup_coords(UNSUP_PATH)
    print(f"Unsupervised: {len(unsup_coords)} particles")

    draw_overlay(gt_coords, vol, Z_SLICE, SIZE_NM, VOXEL_SPACING,
                 'lime', 'Ground Truth',
                 os.path.join(OUTPUT_DIR, 'TS_0008_z250_ground_truth.png'))

    draw_overlay(unsup_coords, vol, Z_SLICE, SIZE_NM, VOXEL_SPACING,
                 'yellow', 'Unsupervised',
                 os.path.join(OUTPUT_DIR, 'TS_0008_z250_unsupervised.png'))

    draw_combined_overlay(gt_coords, unsup_coords, vol, Z_SLICE, SIZE_NM, VOXEL_SPACING,
                          os.path.join(OUTPUT_DIR, 'TS_0008_z250_combined.png'))

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
