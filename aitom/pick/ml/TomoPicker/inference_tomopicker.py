# This script is originally written by Md. Zarif Ul Alam, later modified by Ajmain Yasar Ahmed
import argparse
import numpy as np
import os
import pandas as pd
import torch
import gc
from scipy.ndimage import generate_binary_structure, maximum_filter, binary_erosion

from train_tomopicker import load_model, load_tomogram

def generate_subtomograms_starting_coords(tomogram, subtomogram_size):
    z_max, y_max, x_max = tomogram.shape
    subtomograms, starting_coords = [], []

    for iz in range(0, z_max, subtomogram_size):
        for iy in range(0, y_max, subtomogram_size):
            for ix in range(0, x_max, subtomogram_size):
                # Extracting subtomograms in a sliding-window manner
                # Padding subtomogram with zero if size mismatches
                if iz + subtomogram_size > z_max or iy + subtomogram_size > y_max or ix + subtomogram_size > x_max:
                    subtomogram = np.zeros(shape=(subtomogram_size, subtomogram_size, subtomogram_size), dtype=np.float32)
                    temp_subtomogram = tomogram[iz:iz + subtomogram_size, iy:iy + subtomogram_size, ix:ix + subtomogram_size]
                    subtomogram[:temp_subtomogram.shape[0], :temp_subtomogram.shape[1], :temp_subtomogram.shape[2]] = temp_subtomogram
                else:
                    subtomogram = tomogram[iz:iz + subtomogram_size, iy:iy + subtomogram_size, ix:ix + subtomogram_size]

                # Storing subtomograms with starting coordinates
                starting_coord = (iz, iy, ix)
                subtomograms.append(subtomogram)
                starting_coords.append(starting_coord)

    return subtomograms, starting_coords

def merge_score_matrix(score_matrix, starting_coords, subtomogram_size, tomogram_shape):
    z_max, y_max, x_max = tomogram_shape
    merged_score_matrix = np.zeros(shape=tomogram_shape, dtype=np.float32)

    assert len(score_matrix) == len(starting_coords)

    # Looping over score_matrix
    for index in range(len(score_matrix)):
        score_matrix_instance, starting_coord = score_matrix[index], starting_coords[index]
        iz, iy, ix = starting_coord

        # Padding score_matrix_instance with -inf if size mismatches
        if iz + subtomogram_size > z_max or iy + subtomogram_size > y_max or ix + subtomogram_size > x_max:
            temp_score_matrix = np.full(shape=(subtomogram_size, subtomogram_size, subtomogram_size), fill_value=-np.inf, dtype=np.float32)
            temp_score_matrix[:score_matrix_instance.shape[0], :score_matrix_instance.shape[1], :score_matrix_instance.shape[2]] = score_matrix_instance
            score_matrix_instance = temp_score_matrix

        # Adding score_matrix_instance to merged_score_matrix
        if iz + subtomogram_size > z_max or iy + subtomogram_size > y_max or ix + subtomogram_size > x_max:
            merged_score_matrix[iz:iz + subtomogram_size, iy:iy + subtomogram_size, ix:ix + subtomogram_size] += score_matrix_instance[:z_max - iz, :y_max - iy, :x_max - ix]
        else:
            merged_score_matrix[iz:iz + subtomogram_size, iy:iy + subtomogram_size, ix:ix + subtomogram_size] += score_matrix_instance

    return merged_score_matrix

def non_maximum_suppression_3D(score_matrix, particle_radius, num_pick_particles=1000, threshold=0):
    z_max, y_max, x_max = score_matrix.shape
    estimations, r = [], particle_radius

    for _ in range(num_pick_particles):
        max_score_position = np.unravel_index(indices=np.argmax(a=score_matrix), shape=score_matrix.shape)
        print(score_matrix[max_score_position])
        #if score_matrix[max_score_position] <= threshold:
        #    break

        z, y, x = max_score_position
        estimations.append([x, y, z, score_matrix[max_score_position]])

        # Masking a cuboidal region around max_score_position in score_matrix with -inf
        score_matrix[max(0, z - r):min(z_max, z + r), max(0, y - r):min(y_max, y + r), max(0, x - r):min(x_max, x + r)] = -np.inf

    return estimations

def generate_estimations_nms3D(tomogram_name, score_matrix, particle_radius, num_pick_particles=1000):
    estimations = non_maximum_suppression_3D(score_matrix=score_matrix, particle_radius=particle_radius, num_pick_particles=num_pick_particles)
    estimations_df = pd.DataFrame(
        {
            "tomogram_name": [tomogram_name] * len(estimations),
            "x_coord": [item[0] for item in estimations],
            "y_coord": [item[1] for item in estimations],
            "z_coord": [item[2] for item in estimations],
            "score": [item[3] for item in estimations]
        }
    )
    return estimations, estimations_df

def get_args():
    parser = argparse.ArgumentParser(description="Python script for picking macromolecules from cryo-electron tomograms using a trained Pumpkin model.")
    metavar = 'X'

    parser.add_argument("--tomograms_path", default=None, type=str, metavar=metavar, help="Path to a folder containing tomograms for inference (Default: None)", dest="tomograms_path")
    parser.add_argument("--tomogram_name", default=None, type=str, metavar=metavar, help="Name of the tomogram from which particles will be picked (Default: None)", dest="tomogram_name")

    parser.add_argument("--encoder", default="basic", type=str, metavar=metavar, help="Type of feature extractor (either basic or yopo) to use in network (Default: basic)", dest="encoder_mode")
    parser.add_argument("--decoder", action="store_true", help="Whether to use sample reconstructor in network (Default: False)", dest="use_decoder")

    parser.add_argument("--size", default=32, type=int, metavar=metavar, help="Size of subtomograms (either 16 or 32) in each dimension (Default: 16)", dest="subtomogram_size")
    parser.add_argument("--radius", default=12, type=int, metavar=metavar, help="Radius of a particle (in voxel) in tomograms (Default: 7)", dest="particle_radius")
    parser.add_argument("--pick", default=2000, type=int, metavar=metavar, help="Number of particles to pick from a tomogram (Default: 1000)", dest="num_pick_particles")

    parser.add_argument("--name", default="pumpkin", type=str, metavar=metavar, help="Name of the saved Pumpkin model (Default: pumpkin)", dest="model_name")
    parser.add_argument("--model_path", default=None, type=str, metavar=metavar, help="Path to a folder containing saved model weights (Default: None)", dest="model_path")

    parser.add_argument("--nms", action="store_true", help="Whether to use NMS algorithm for particle picking (Default: False)", dest="use_nms")

    parser.set_defaults(use_decoder=False)
    parser.set_defaults(use_nms=False)

    args = parser.parse_args()

    encoder_modes = ["basic", "unet"]

    assert args.particle_radius > 0, "Invalid particle_radius provided!"
    assert args.num_pick_particles > 0, "Invalid num_pick_particles provided!"

    return args

if __name__ == "__main__":
    args = get_args()

    # Loading query tomogram and preparing subtomograms from it for particles extraction
    tomogram = load_tomogram(tomogram_path=f"{args.tomograms_path}/{args.tomogram_name}")
    tomogram_shape = tomogram.shape

    subtomograms, starting_coords = generate_subtomograms_starting_coords(tomogram=tomogram, subtomogram_size=args.subtomogram_size)
    print(f"Number of Generated Subtomograms: {len(subtomograms)}")

    del tomogram
    gc.collect()

    # Loading a trained Pumpkin model and estimating prediction scores for subtomograms
    subtomograms = torch.from_numpy(np.array(subtomograms, dtype=np.float32))
    subtomograms = torch.unsqueeze(subtomograms, dim=1)

    pumpkin_model = load_model(args=args)

    if torch.cuda.is_available():
        pumpkin_model = pumpkin_model.cuda()
        subtomograms = subtomograms.cuda()

    print(subtomograms.shape)
    prediction_scores = torch.zeros_like(subtomograms)
    print(prediction_scores.shape)
    with torch.no_grad():
        for i in range(0,subtomograms.shape[0],100):
            print('Iteration',i)
            score, _  = pumpkin_model(subtomograms[i:i+100]) 
            prediction_scores[i:i+100] = score.clone()
            del score
            torch.cuda.empty_cache()
    
    #prediction_scores, _ = pumpkin_model(subtomograms)

    del subtomograms, _
    gc.collect()

    # Merging prediction scores for subtomograms to get global prediction scores for query tomogram
    prediction_scores_matrix = np.squeeze(a=prediction_scores.detach().cpu().numpy(), axis=1)
    prediction_scores_matrix = merge_score_matrix(score_matrix=prediction_scores_matrix, starting_coords=starting_coords, subtomogram_size=args.subtomogram_size, tomogram_shape=tomogram_shape)

    del prediction_scores, starting_coords
    gc.collect()

    # Extracting estimated particles centers from global prediction scores for query tomogram
    estimations, estimations_df = generate_estimations_nms3D(
            tomogram_name=args.tomogram_name,
            score_matrix=prediction_scores_matrix,
            particle_radius=args.particle_radius,
            num_pick_particles=args.num_pick_particles
        )
    print(f"Number of Picked Particles (3D NMS): {len(estimations)}")
    
    del prediction_scores_matrix, estimations
    gc.collect()

    estimations_path = f"estimations/{args.model_name}"

    if not os.path.exists(estimations_path):
        os.makedirs(estimations_path)

    estimations_df.to_csv(f"{estimations_path}/estimations_{args.tomogram_name}_{args.num_pick_particles}.csv", index=False)