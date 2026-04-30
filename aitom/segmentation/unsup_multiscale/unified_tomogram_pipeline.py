#!/usr/bin/env python3
"""
Unified Tomogram Processing Pipeline

This script combines all the individual processing steps into a single configurable pipeline
that can process multiple tomograms with different slice ranges and parameters.

Steps:
1. Extract slices from MRC files
2. Split slices into patches (2x2 grid)
3. Extract stable diffusion eigenvectors
4. Create canvas images from eigenvectors
5. Correct canvas images using neighboring slices
6. Create membrane masks using CellPose
7. Extract macromolecule coordinates and save to CSV

Usage:
    python unified_tomogram_pipeline.py [arguments]
"""

import cv2
import os
import numpy as np
import mrcfile
import torch
import pickle
import time
import pandas as pd
import math
import shutil
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from model.diffusion_extractor import StableDiffusion
from model.vit_extractor import MAEFeatureExtractor, DINOFeatureExtractor
from train_utils import ncut_loss, ToyCNN, SimpleDataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.measure import regionprops, label
from skimage.transform import rescale, resize
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, gaussian_filter
from cellpose import models
from sklearn.preprocessing import minmax_scale

# Set thread limits for optimal performance
default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"


class TomogramProcessor:
    """Main class for processing tomograms through the complete pipeline."""
    
    def __init__(self, args, processing_params=None):
        # Use processing parameters from config or defaults
        if processing_params:
            self.voxel_spacing_in_nm = processing_params.get('voxel_spacing_in_nm', 1.348)
            self.max_particle_diameter_in_nm = processing_params.get('max_particle_diameter_in_nm', 30)
            self.flow_threshold = processing_params.get('flow_threshold', 0.4)
            self.cellprob_threshold = processing_params.get('cellprob_threshold', 0.0)
            self.circularity_threshold = processing_params.get('circularity_threshold', 0.95)
            self.dist_thresh = processing_params.get('dist_thresh', 10)
            self.pixel_threshold = processing_params.get('pixel_threshold', 100)
            self.area_threshold = processing_params.get('area_threshold', 0.4)
        else:
            # Default values
            self.voxel_spacing_in_nm = 1.348
            self.max_particle_diameter_in_nm = 30
            self.flow_threshold = 0.4
            self.cellprob_threshold = 0.0
            self.circularity_threshold = 0.95
            self.dist_thresh = 10
            self.pixel_threshold = 100
            self.area_threshold = 0.4
        
        # Set noise_max_t based on voxel spacing
        args.noise_max_t = int(68 // self.voxel_spacing_in_nm)
        print(f"Setting noise_max_t to {args.noise_max_t} based on voxel spacing {self.voxel_spacing_in_nm} nm")
        
        self.args = args
        self.attention_extractor = StableDiffusion(args)
        self.cellpose_model = models.CellposeModel(gpu=True)
    
    def contrast_stretching(self, image: np.ndarray) -> np.ndarray:
        """Calculate the minimum and maximum pixel values and stretch contrast."""
        stretched_image = 255 * minmax_scale(image.flatten()).reshape(image.shape)
        
        # Convert to uint8 format
        stretched_image = stretched_image.astype(np.uint8)
        
        return stretched_image
    
    def process_slabs(self, image: np.ndarray) -> np.ndarray:
        """Process slice data with contrast stretching, CLAHE, and smoothing."""
        stretched_image = self.contrast_stretching(image)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        clahed_image = clahe.apply(stretched_image)
        smoothed_image = self.contrast_stretching(gaussian_filter(stretched_image, sigma=1.25))
        final_image = Image.fromarray(smoothed_image, 'L')
        return final_image
        
    def load_mrc_and_save_slices(self, mrc_file_path, output_slice_dir, slab_range_start=None, slab_range_end=None):
        """
        Load MRC file in zyx format and save slices in the specified range.
        
        Args:
            mrc_file_path: Path to the MRC file
            output_slice_dir: Directory to save the slices
            slab_range_start: Starting slice index (default: 0)
            slab_range_end: Ending slice index (default: last slice)
        """
        os.makedirs(output_slice_dir, exist_ok=True)
        
        with mrcfile.open(mrc_file_path, mode='r') as mrc:
            data = np.array(mrc.data)
            
            if data is None:
                raise ValueError(f"Could not load data from MRC file: {mrc_file_path}")
            
            z_dim, y_dim, x_dim = data.shape
            
            if slab_range_start is None:
                slab_range_start = 0
            if slab_range_end is None:
                slab_range_end = z_dim
                
            slab_range_start = max(0, slab_range_start)
            slab_range_end = min(z_dim, slab_range_end)
            
            print(f"MRC file shape: {data.shape}")
            print(f"Saving slices {slab_range_start} to {slab_range_end-1}")
            
            for z in range(slab_range_start, slab_range_end):
                print(f"Saving slice {z}")
                slice_data = data[z]

                slice_filename = f"slice_{z:04d}.png"
                slice_path = os.path.join(output_slice_dir, slice_filename)
                processed_image = self.process_slabs(slice_data)
                processed_image.save(slice_path)
                
            print(f"Saved {slab_range_end - slab_range_start} slices to {output_slice_dir}")

    def split_image_directory(self, input_dir, output_dir_4patches):
        """Split images into 4 patches (2x2 grid) and resize to 512x512."""
        os.makedirs(output_dir_4patches, exist_ok=True)
        
        for image_file in os.listdir(input_dir):
            image_path = os.path.join(input_dir, image_file)

            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            image = cv2.imread(image_path)
            img_name = os.path.splitext(image_file)[0]

            resized_image_1024x1024 = cv2.resize(image, (1024, 1024))
            
            height, width = resized_image_1024x1024.shape[:2]
            half_height, half_width = height // 2, width // 2

            patches_4 = [
                resized_image_1024x1024[:half_height, :half_width],      # Top-left
                resized_image_1024x1024[:half_height, half_width:],      # Top-right
                resized_image_1024x1024[half_height:, :half_width],      # Bottom-left
                resized_image_1024x1024[half_height:, half_width:],      # Bottom-right
            ]

            for i, patch in enumerate(patches_4):
                row = i // 2
                col = i % 2
                patch_name = f'{img_name}_{row}_{col}.jpg'
                cv2.imwrite(os.path.join(output_dir_4patches, patch_name), patch)
                
            print(f"4 patches saved for {image_file} in {output_dir_4patches}")

    def save_stable_diffusion_eigenvectors(self, image_dir, eigen_pickle_dir, slab_range_start, slab_range_end):
        """Extract and save stable diffusion eigenvectors for specified slice range."""
        os.makedirs(eigen_pickle_dir, exist_ok=True)
        
        files = os.listdir(image_dir)
        print('Total files', len(files))
        
        process_list = []
        for file in files:
            attribute_names = file.split('.')[0].split('_')
            z_index = int(attribute_names[-3])
            if slab_range_start <= z_index < slab_range_end:
                process_list.append(file)

        print(f"Processing {len(process_list)} files for eigenvectors")
        
        train_steps = self.args.train_steps
        for num, img_name in tqdm(enumerate(process_list), desc="Extracting eigenvectors"):
            img_path = os.path.join(image_dir, img_name)
            out_path = os.path.join(eigen_pickle_dir, img_name.split(".")[0] + '.pkl')
            if os.path.exists(out_path):
                continue
            
            # Convert grayscale to RGB
            gray = Image.open(img_path).convert('L')
            rgb_gray = Image.merge("RGB", (gray, gray, gray))
            rgb_img = np.array(rgb_gray)
            
            img = (torch.from_numpy(rgb_img).float() / 255)[None,...].permute(0,3,1,2)
            img = img.to(self.args.gpu)
            input_feats = self.attention_extractor.process_input(img)
            toy_cnn = ToyCNN(self.attention_extractor.feat_h, self.attention_extractor.feat_w, int(self.args.num_of_eig))
            toy_cnn.to(self.args.gpu)
            optim = torch.optim.Adam(toy_cnn.parameters(), lr=self.args.learning_rate)

            # Main optimization loop
            for i in range(train_steps + 1):
                ortho_list = []
                eigval_list = []
                attns = self.attention_extractor.collect_attention(input_feats, i)
                dense_feat = toy_cnn().softmax(dim=1)
                for attn in attns:
                    eigval, ortho = ncut_loss(attn, dense_feat, self.args.symmetric_matrix)
                    eigval_list.append(eigval)
                    ortho_list.append(ortho)
                eigval = torch.cat(eigval_list).reshape(-1)
                ortho = torch.cat(ortho_list).reshape(-1)
                loss = self.args.eig_loss_weight * (eigval - (1 - self.args.symmetric_matrix)).abs().mean() + self.args.ortho_loss_weight * ortho.pow_(2).mean()
                loss.backward()
                if i % self.args.accum_grads == 0 and i != 0:
                    optim.step()
                    optim.zero_grad()
            
            # Release the attention hook
            self.attention_extractor.clear_after_loop()

            dense_feat = toy_cnn().softmax(dim=1)
            u, s, v = F.normalize(dense_feat.flatten(2, -1), dim=-1)[0].T.svd()
            reorth_dense_feat = torch.mm(u, torch.diag(s)).T
            dense_feat = reorth_dense_feat.reshape(dense_feat.shape)
            eigenvector = u.T.reshape(dense_feat.shape)
            with open(out_path, 'wb') as f:
                pickle.dump(eigenvector.cpu().detach().numpy().squeeze(), f)

    def conditional_invert(self, gray_image, pixel_threshold, area_threshold):
        """Conditionally invert image based on pixel intensity distribution."""
        above_threshold = np.sum(gray_image > pixel_threshold)
        total_pixels = gray_image.size
        ratio = above_threshold / total_pixels
        if ratio < area_threshold:
            gray_image = 255 - gray_image
        return gray_image

    def diversity_score(self, image, patch_size=4, overlap=2):
        """Calculate diversity score of image using patch variance."""
        stride = patch_size - overlap
        inten_list = []

        for i in range(0, image.shape[0] - patch_size + 1, stride):
            for j in range(0, image.shape[1] - patch_size + 1, stride):
                patch = image[i:i+patch_size, j:j+patch_size]
                inten_list.append(np.std(patch))

        return np.std(inten_list)

    def count_contours(self, gray_img):
        """Count contours in grayscale image using Otsu thresholding."""
        if gray_img.dtype != np.uint8:
            gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
            gray_img = gray_img.astype(np.uint8)
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)

    def select_best_eigen(self, dat):
        """Select best eigenvector based on variance and contour count."""
        index = -1
        maxVar = -np.inf
        for i in range(len(dat)):
            cc = self.count_contours(dat[i])
            if cc > 10:
                continue
            var = self.diversity_score(dat[i])
            if var > maxVar:
                maxVar = max(var, maxVar)
                index = i
        return dat[index] if index >= 0 else None, index

    def merge_eigens(self, base, dat, candidates):
        """Merge eigenvectors based on PSNR and SSIM similarity metrics."""
        maxPSNR = -np.inf
        bestCandidate = None
        bestIndex = -1
        merged = None
        for i in range(len(dat)):
            if i not in candidates:
                continue
            item = dat[i]
            psnr_value = psnr(base, item, data_range=base.max() - base.min())
            ssim_value = ssim(base, item, data_range=base.max() - base.min())
            if ssim_value > 0.35 and psnr_value > maxPSNR:
                arr = base + item
                arr = (arr - np.min(arr)) * 255 / (np.max(arr) - np.min(arr))
                if self.count_contours(arr) > 15:
                    continue
                if self.diversity_score(arr) < self.diversity_score(base):
                    continue
                maxPSNR = psnr_value
                bestCandidate = item
                merged = arr
                bestIndex = i
        return merged, bestCandidate, bestIndex

    def create_canvas_images(self, input_path, output_path, tomogram_name):
        """Create canvas images from eigenvector pickles by selecting and merging best eigenvectors."""
        os.makedirs(output_path, exist_ok=True)
        files = os.listdir(input_path)

        for i, file in enumerate(files):
            f_path = os.path.join(input_path, file)
            with open(f_path, 'rb') as f:
                dat = pickle.load(f)
            dat = dat[:10]  # Use first 10 eigenvectors
            
            # Normalize each eigenvector
            for j in range(len(dat)):
                dat[j] = (dat[j] - np.min(dat[j])) * 255 / (np.max(dat[j]) - np.min(dat[j]))
                dat[j] = self.conditional_invert(dat[j], self.pixel_threshold, self.area_threshold)

            candidates = [k for k in range(len(dat))]
            base, index = self.select_best_eigen(dat)
            if index == -1 or base is None:
                continue
            candidates.remove(index)
            
            # Merge up to 10 eigenvectors
            for k in range(10):
                merged, bestCandidate, bestIndex = self.merge_eigens(base, dat, candidates)
                if merged is None:
                    break
                candidates.remove(bestIndex)
                base = merged
                
            base = base.astype(np.uint8)
            resized_base_image = cv2.resize(base, (512, 512), interpolation=cv2.INTER_CUBIC)
            
            # Invert the image if needed
            final_image = self.conditional_invert(resized_base_image, self.pixel_threshold, self.area_threshold)
            #final_image = resized_base_image
            
            # Correctly format the output filename with the tomogram name
            base_name = os.path.splitext(file)[0]
            # The base_name is like 'slice_0200_0_0', we need to replace 'slice' with tomogram_name
            parts = base_name.split('_')
            output_filename = f"{tomogram_name}_{'_'.join(parts[1:])}.png"

            output_file_path = os.path.join(output_path, output_filename)
            cv2.imwrite(output_file_path, final_image)
            
        print(f"Canvas images created in {output_path}")

    def correct_canvas_images(self, tomogram_name, input_path, output_path, slab_start_index, slab_end_index):
        """Correct canvas images using neighboring slice information via SSIM analysis."""
        os.makedirs(output_path, exist_ok=True)
        files = os.listdir(input_path)

        for i, file in enumerate(files):
            attribute_names = file.split('.')[0].split('_')
            z_index = attribute_names[-3]
            if int(z_index) > slab_end_index or int(z_index) < slab_start_index:
                continue
            row = attribute_names[-2]
            col = attribute_names[-1]
            img_name = '_'.join([tomogram_name, z_index, row, col])
            prev_img_name = '_'.join([tomogram_name, str(int(z_index)-1), row, col])
            next_img_name = '_'.join([tomogram_name, str(int(z_index)+1), row, col])
            
            img_path = os.path.join(input_path, img_name + '.png')
            prev_img_path = os.path.join(input_path, prev_img_name + '.png')
            next_img_path = os.path.join(input_path, next_img_name + '.png')

            current_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if os.path.exists(prev_img_path) and os.path.exists(next_img_path):
                prev_img = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE)
                next_img = cv2.imread(next_img_path, cv2.IMREAD_GRAYSCALE)
                prev_curr_ssim = ssim(prev_img, current_img, data_range=prev_img.max() - prev_img.min())
                next_curr_ssim = ssim(next_img, current_img, data_range=prev_img.max() - prev_img.min())
                prev_next_ssim = ssim(prev_img, next_img, data_range=prev_img.max() - prev_img.min())
                
                if prev_next_ssim > 0.90 and prev_curr_ssim < 0.8 and next_curr_ssim < 0.8:
                    # Replace current with average of neighbors
                    img = np.array(prev_img).astype(np.float32) * 0.5 + np.array(next_img).astype(np.float32) * 0.5
                    img = (img - np.min(img)) * 255 / (np.max(img) - np.min(img))
                else:
                    # Blend current with neighbors
                    img = np.array(current_img).astype(np.float32) + np.array(prev_img).astype(np.float32) * 0.5 + np.array(next_img).astype(np.float32) * 0.5
                    img = (img - np.min(img)) * 255 / (np.max(img) - np.min(img))
            else:
                img = current_img.astype(np.float32)
                
            img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(output_path, img_name + '.png'), img)

    def create_membrane_masks(self, input_path, patch_output_path, slab_start_index, slab_end_index, tomogram_size=(960, 928), merged_output_path=None):
        """Create membrane masks from canvas images using adaptive thresholding and CellPose."""
        os.makedirs(patch_output_path, exist_ok=True)
        if merged_output_path:
            os.makedirs(merged_output_path, exist_ok=True)
        files = os.listdir(input_path)

        processed_masks = []
        mask_names = []
        
        for i, file in enumerate(files):
            z_index = file.split('.')[0].split('_')[-3]
            if int(z_index) >= slab_end_index or int(z_index) < slab_start_index:
                continue
            img_path = os.path.join(input_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply Gaussian blur and adaptive thresholding
            img = cv2.GaussianBlur(img, (7, 7), 1.5)
            binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 3)
            inverted_mask = cv2.bitwise_not(binary)
            
            processed_masks.append(inverted_mask)
            mask_names.append(file.split('.')[0] + '_mask.png')

        if len(processed_masks) == 0:
            print("No images found in the specified slice range for membrane mask creation.")
            return

        print(f"Processing {len(processed_masks)} masks with CellPose...")
        cellpose_masks, _, _ = self.cellpose_model.eval(processed_masks, diameter=10, 
                                                       flow_threshold=self.flow_threshold, 
                                                       cellprob_threshold=self.cellprob_threshold)

        for i in range(len(mask_names)):
            cellpose_mask = cellpose_masks[i]
            edges = processed_masks[i]
            
            # Create ribosome mask with circularity filtering and erosion
            ribosome_mask = np.zeros_like(cellpose_mask, dtype=np.uint8)
            for region in regionprops(cellpose_mask):
                area = region.area
                coords = region.coords
                perimeter = region.perimeter if region.perimeter > 0 else 1
                circularity = 4 * np.pi * region.area / (perimeter ** 2)
                if circularity >= self.circularity_threshold:
                    ribosome_mask[tuple(coords.T)] = 255
            
            # Create final membrane mask
            membrane_mask = np.bitwise_and(edges, np.bitwise_not(ribosome_mask))
                    
            cv2.imwrite(os.path.join(patch_output_path, mask_names[i]), membrane_mask.astype(np.uint8))
        
                # Step 6b: Merge membrane masks for each slice
        if merged_output_path:
            print("Merging membrane mask patches into full-sized masks...")
            
            # Get unique z-indices from processed masks
            z_indices = set()
            for mask_name in mask_names:
                z_index = int(mask_name.split('_')[-4])  # Extract z-index from filename
                z_indices.add(z_index)
            
            # Merge masks for each z-index
            for z_index in sorted(z_indices):
                # Extract tomogram name from first mask file
                tomogram_name = '_'.join(mask_names[0].split('_')[:2]) # Get tomogram name from first file
                print(tomogram_name)
                merged_mask = self.merge_membrane_masks(
                    membrane_masks_path=patch_output_path,
                    tomogram_name=tomogram_name,
                    z_index=z_index,
                    tomogram_size=tomogram_size
                )
                
                if merged_mask is not None:
                    merged_filename = f"{tomogram_name}_{z_index:04d}_membrane_mask.png"
                    merged_path = os.path.join(merged_output_path, merged_filename)
                    cv2.imwrite(merged_path, merged_mask)
                    print(f"Saved merged mask: {merged_filename}")
                else:
                    print(f"Failed to merge masks for z-index {z_index}")
            
            print(f"Merged membrane masks saved to: {merged_output_path}")

    def remove_all_close_pairs(self, points, min_distance):
        """Remove all points that are within min_distance of any other point."""
        if len(points) == 0:
            return points
        tree = cKDTree(points)
        to_discard = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if to_discard[i]:
                continue
            neighbors = tree.query_ball_point(points[i], r=min_distance)
            neighbors = [j for j in neighbors if j != i]
            if neighbors:
                to_discard[i] = True
                to_discard[neighbors] = True

        return points[~to_discard]

    def retain_unique_points(self, points, threshold):
        """Retain only one point among close neighbors within a given threshold."""
        if len(points) == 0:
            return points
        tree = cKDTree(points)
        taken = np.zeros(len(points), dtype=bool)
        keep_indices = []

        for i in range(len(points)):
            if not taken[i]:
                neighbors = tree.query_ball_point(points[i], r=threshold)
                taken[neighbors] = True
                keep_indices.append(i)
        
        return points[keep_indices]

    def merge_membrane_masks(self, membrane_masks_path, tomogram_name, z_index, tomogram_size=(960, 928)):
        """
        Merge 4 membrane mask patches (2x2 grid) into a single full-sized mask.
        
        Args:
            membrane_masks_path: Path to the membrane masks directory
            tomogram_name: Name of the tomogram
            z_index: Z slice index
            tomogram_size: Target size for the final mask (width, height)
            
        Returns:
            merged_mask: Full-sized membrane mask resized to tomogram dimensions
        """
        # Define the patch filenames
        print(tomogram_name)
        patch_files = [
            f"{tomogram_name}_{z_index:04d}_0_0_mask.png",  # top-left
            f"{tomogram_name}_{z_index:04d}_0_1_mask.png",  # top-right
            f"{tomogram_name}_{z_index:04d}_1_0_mask.png",  # bottom-left
            f"{tomogram_name}_{z_index:04d}_1_1_mask.png"   # bottom-right
        ]
        
        patches = []
        missing_files = []
        
        # Load all 4 patches
        print(membrane_masks_path)
        print(patch_files)
        
        for patch_file in patch_files:
            patch_path = os.path.join(membrane_masks_path, patch_file)
            if os.path.exists(patch_path):
                patch = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
                patches.append(patch)
            else:
                missing_files.append(patch_file)
                patches.append(None)
        
        # Check if all patches are available
        if len(missing_files) > 0:
            print(f"Warning: Missing patch files for z={z_index}: {missing_files}")
            return None
        
        # Get patch dimensions (assuming all patches are same size, typically 512x512)
        patch_height, patch_width = patches[0].shape[:2]
        
        # Create the 1024x1024 merged image
        merged_mask = np.zeros((patch_height * 2, patch_width * 2), dtype=np.uint8)
        
        # Arrange patches in 2x2 grid
        # Top-left (0,0)
        merged_mask[0:patch_height, 0:patch_width] = patches[0]
        # Top-right (0,1)  
        merged_mask[0:patch_height, patch_width:patch_width*2] = patches[1]
        # Bottom-left (1,0)
        merged_mask[patch_height:patch_height*2, 0:patch_width] = patches[2]
        # Bottom-right (1,1)
        merged_mask[patch_height:patch_height*2, patch_width:patch_width*2] = patches[3]
        
        # Resize to tomogram size
        final_mask = cv2.resize(merged_mask, tomogram_size, interpolation=cv2.INTER_NEAREST)
        
        return final_mask

    def extract_macromolecule_coordinates(self, canvas_path, membrane_path, tomogram_name, tomogram_size=(960, 928), output_dir='.'):
        """Extract macromolecule coordinates from processed images and save to CSV."""
        files = os.listdir(canvas_path)
        print(canvas_path)
        print(membrane_path)
        global_coords = []
        for i, file in enumerate(files):
            attribute_names = file.split('.')[0].split('_')
            z_index = int(attribute_names[-3])
            row = int(attribute_names[-2])
            col = int(attribute_names[-1])   
            
            img_path = os.path.join(canvas_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
   
            img = cv2.GaussianBlur(img, (7, 7), 1.5)
            binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 3)
            inverted_mask = cv2.bitwise_not(binary)
            
            
            cellpose_mask, _, _ = self.cellpose_model.eval(inverted_mask, diameter=10, flow_threshold=self.flow_threshold, 
                                                          cellprob_threshold=self.cellprob_threshold)
            
            ribosome_mask = np.zeros_like(cellpose_mask, dtype=np.uint8)
            points = []
            refined_points = []
            
            # Check for membrane mask
            edges_path = os.path.join(membrane_path, file.replace('.png', '_mask.png'))
            if os.path.exists(edges_path):
                print('Edges exist')
                edges = cv2.imread(edges_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                distance_from_edges = distance_transform_edt(~edges)
            else:
                # Default distance if no membrane mask
                distance_from_edges = np.ones_like(binary) * self.dist_thresh * 2
            
            for region in regionprops(cellpose_mask):
                area = region.area
                coords = region.coords
                perimeter = region.perimeter if region.perimeter > 0 else 1
                circularity = 4 * np.pi * region.area / (perimeter ** 2)
                
                if circularity >= self.circularity_threshold and area >= (30 / self.voxel_spacing_in_nm):
                    ribosome_mask[tuple(coords.T)] = 255
                    y, x = region.centroid
                    if distance_from_edges[int(y), int(x)] > self.dist_thresh:
                        points.append([x, y])
            
            if len(points) > 0:
                refined_points = self.remove_all_close_pairs(np.array(points), min_distance=32)
            
            # Convert patch coordinates to global coordinates
            for pt in refined_points:
                x, y = pt[0], pt[1]
                x = x * ((tomogram_size[0] // 2) / 512)
                y = y * ((tomogram_size[1] // 2) / 512)
                global_x = x + col * (tomogram_size[0] // 2)
                global_y = y + row * (tomogram_size[1] // 2)
                global_coords.append((int(global_x), int(global_y), z_index))
        
        if len(global_coords) > 0:
            locations = self.retain_unique_points(np.array(global_coords), 
                                                threshold=math.ceil(15 / self.voxel_spacing_in_nm))
            d = {
                'x_coord': list(locations[:, 0]),
                'y_coord': list(locations[:, 1]),
                'z_coord': list(locations[:, 2])
            }
            
            df = pd.DataFrame(d)
            csv_filename = f'{tomogram_name}_unsup_coords_membrane_filtered.csv'
            csv_path = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(locations)} coordinates to {csv_path}")
        else:
            print(f"No coordinates found for {tomogram_name}")

    def process_tomogram(self, tomogram_name, config, input_path, output_path):
        """Process a single tomogram through the entire pipeline."""
        print(f"\n{'='*50}")
        print(f"Processing {tomogram_name}")
        print(f"{'='*50}")
        
        slab_range_start = config['slab_range_start']
        slab_range_end = config['slab_range_end']
        
        # Create directory structure
        base_output = os.path.join(output_path, tomogram_name)
        slices_dir = os.path.join(base_output, 'slices')
        patches_dir = os.path.join(base_output, 'patches_4')
        eigens_dir = os.path.join(base_output, 'eigenvectors')
        canvas_dir = os.path.join(base_output, 'canvas')
        canvas_corrected_dir = os.path.join(base_output, 'canvas_corrected')
        patch_membrane_mask_dir = os.path.join(base_output, 'patch_membrane_masks')
        membrane_mask_dir = os.path.join(base_output, 'membrane_masks')
        macromolecule_coords_dir = os.path.join(base_output, 'macromolecule_coordinates')
        
        for dir_path in [slices_dir, patches_dir, eigens_dir, canvas_dir, canvas_corrected_dir, patch_membrane_mask_dir, membrane_mask_dir, macromolecule_coords_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            # Step 1: Extract slices from MRC
            mrc_file_path = os.path.join(input_path, f'{tomogram_name}.mrc')
            print("\nStep 1: Extracting slices from MRC file...")
            self.load_mrc_and_save_slices(mrc_file_path, slices_dir, slab_range_start, slab_range_end)
            
            # Step 2: Split into patches
            print("\nStep 2: Splitting slices into patches...")
            self.split_image_directory(slices_dir, patches_dir)
            
            # Step 3: Extract stable diffusion eigenvectors
            print("\nStep 3: Extracting stable diffusion eigenvectors...")
            self.save_stable_diffusion_eigenvectors(patches_dir, eigens_dir, slab_range_start, slab_range_end)
            
            # Step 4: Create canvas images
            print("\nStep 4: Creating canvas images...")
            self.create_canvas_images(
                eigens_dir,
                canvas_dir,
                tomogram_name
            )
            
            # Step 5: Correct canvas images
            print("\nStep 5: Correcting canvas images...")
            self.correct_canvas_images(tomogram_name, canvas_dir, canvas_corrected_dir, slab_range_start, slab_range_end)
            
            # Step 6: Create membrane masks
            print("\nStep 6: Creating membrane masks...")
            tomogram_size = config.get('tomogram_size', (960, 928))
            self.create_membrane_masks(canvas_corrected_dir, patch_membrane_mask_dir, slab_range_start, slab_range_end, tomogram_size, membrane_mask_dir)
            
            # Step 7: Extract macromolecule coordinates
            print("\nStep 7: Extracting macromolecule coordinates...")
            tomogram_size = config.get('tomogram_size', (960, 928))
            self.extract_macromolecule_coordinates(canvas_corrected_dir, patch_membrane_mask_dir, tomogram_name, tomogram_size, macromolecule_coords_dir)
            
            print(f"\n✓ Completed processing {tomogram_name}")
            
            # Cleanup: Remove intermediate directories, keep only essential outputs
            print("\nStep 8: Cleaning up intermediate files...")
            
            dirs_to_remove = [slices_dir, patches_dir, eigens_dir, canvas_dir, canvas_corrected_dir, patch_membrane_mask_dir]
            dirs_to_keep = [membrane_mask_dir, macromolecule_coords_dir]
            
            for dir_path in dirs_to_remove:
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Removed directory: {os.path.basename(dir_path)}")
                    except Exception as cleanup_error:
                        print(f"Warning: Could not remove {dir_path}: {cleanup_error}")
            
            print(f"Cleanup complete. Kept directories: {[os.path.basename(d) for d in dirs_to_keep]}")
            
        except Exception as e:
            print(f"\n✗ Error processing {tomogram_name}: {e}")
            raise


def main():
    """Main function to run the unified tomogram processing pipeline."""
    parser = argparse.ArgumentParser(description="Unified Tomogram Processing Pipeline")
    
    # Basic parameters
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--train_steps', type=int, default=2000, help='number of optimization steps')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate for optimization')
    parser.add_argument('--accum_grads', type=int, default=20, help='number of steps to accumulate gradients for')
    parser.add_argument('--model', type=str, default='stable_diffusion', 
                       choices=['stable_diffusion', 'mae', 'dino'], help='model to extract attention')
    parser.add_argument('--huggingface_token', type=str, default='', help='huggingface API token to load stable diffusion')
    
    # Model args
    parser.add_argument('--symmetric_matrix', type=int, default=0, choices=[0,1], 
                       help='1 symmetrical normalized randomwalk matrix, 0 for conventional attention matrix')
    parser.add_argument('--vit_patch_size', type=int, default=8, choices=[8,16], help='the patch size in DINO VIT')
    parser.add_argument('--vit_model_arch', type=str, default='vit_base', 
                       choices=['vit_base', 'vit_small'], help='the model arch in DINO VIT')
    parser.add_argument('--vit_resize_img_size', type=int, default=480, help='the smaller edge of input image for VIT')
    parser.add_argument('--num_registers', type=int, default=0, help="Number of register tokens")
    
    # Noise sampling
    parser.add_argument('--noise_schedule', type=str, default='random', 
                       choices=['random', 'constant', 'increasing', 'decreasing', 'cyclic'], 
                       help='noise schedule to use for training')
    parser.add_argument('--noise_min_t', type=int, default=0, help='minimum t to use in diffusion model')
    parser.add_argument('--noise_max_t', type=int, default=50, help='maximum t to use in diffusion model')
    parser.add_argument('--noise_periods', type=float, default=1, help='periods for cyclic noise schedule')
    parser.add_argument('--noise_sampling', action='store_true', default=True, 
                       help='if true, sample noise random uniformly from below maximum value defined by schedule')
    
    # Eigenvector parameters
    parser.add_argument('--num_of_eig', type=int, default=12, help='number of eigenvector to compute')
    parser.add_argument('--eig_loss_weight', type=float, default=2, help='weight ratio for primary objective')
    parser.add_argument('--ortho_loss_weight', type=float, default=2, help='weight ratio for orthogonal regularization')
    
    # Attention buffer
    parser.add_argument('--use_buffer_prob', type=float, default=0.9, help='chance to use buffer')
    parser.add_argument('--attn_buffer_size', type=int, default=1, help='attn buffer size')
    
    # Path arguments
    parser.add_argument('--input_path', type=str,
                       default='input/',
                       help='Input directory containing MRC files')
    parser.add_argument('--output_path', type=str,
                       default='output/',
                       help='Output directory for processed results')
    
    args = parser.parse_args()
    
    # Import configuration
    try:
        from tomogram_config_testing import tomogram_configs, processing_params, default_paths
        print(f"Loaded configuration for {len(tomogram_configs)} tomograms")
    except ImportError:
        print("Warning: tomogram_config.py not found, using default configuration")
        tomogram_configs = {
            'TS_0008': {
                'slab_range_start': 250,
                'slab_range_end': 253,
                'tomogram_size': (960, 928)
            }
        }
        processing_params = {}
        default_paths = {
            'input_path': 'input/',
            'output_path': 'output/',
        }
    
    # Use paths from config or command line
    if hasattr(args, 'input_path') and args.input_path:
        input_path = args.input_path
    else:
        input_path = default_paths.get('input_path', 'input/')

    if hasattr(args, 'output_path') and args.output_path:
        output_path = args.output_path
    else:
        output_path = default_paths.get('output_path', 'output/')

    print("Unified Tomogram Processing Pipeline")
    print("=" * 50)
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"GPU: {args.gpu}")
    print(f"Training steps: {args.train_steps}")
    print(f"Tomograms to process: {list(tomogram_configs.keys())}")
    
    # Initialize processor
    processor = TomogramProcessor(args, processing_params)
    
    # Process each tomogram
    successful_tomograms = []
    failed_tomograms = []
    
    for tomogram_name, config in tomogram_configs.items():
        try:
            processor.process_tomogram(tomogram_name, config, input_path, output_path)
            successful_tomograms.append(tomogram_name)
        except Exception as e:
            print(f"Error processing {tomogram_name}: {e}")
            failed_tomograms.append(tomogram_name)
            continue
    
    # Summary
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Successfully processed: {successful_tomograms}")
    if failed_tomograms:
        print(f"Failed to process: {failed_tomograms}")
    print(f"Total successful: {len(successful_tomograms)}")
    print(f"Total failed: {len(failed_tomograms)}")
    print("=" * 50)


if __name__ == '__main__':
    main() 
