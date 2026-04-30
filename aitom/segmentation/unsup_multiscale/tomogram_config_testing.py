# Tomogram Configuration File
# Define processing parameters for each tomogram

# Processing parameters (these affect the analysis quality)
processing_params = {
    'voxel_spacing_in_nm': 1.348,        # Voxel spacing in nanometers
    'max_particle_diameter_in_nm': 30,   # Maximum expected particle diameter
    'flow_threshold': 0.4,               # CellPose flow threshold
    'cellprob_threshold': 0.0,           # CellPose cell probability threshold
    'circularity_threshold': 0.95,       # Minimum circularity for particles
    'dist_thresh': 10,                   # Distance threshold from membrane
    'pixel_threshold': 100,              # Pixel intensity threshold
    'area_threshold': 0.4               # Area threshold for image inversion
}

# Default file paths
# Set these to your local directories, or override via --input_path / --output_path CLI flags
default_paths = {
    'input_path': 'input/',
    'output_path': 'output/'
}

# Configuration for each tomogram to process
# Add your MRC files here with their processing parameters
tomogram_configs = {
    # Example configuration - replace with your actual tomogram names and parameters
    'TS_0008': {
        'slab_range_start': 250,            # Starting slice index for processing
        'slab_range_end': 253,              # Ending slice index for processing
        'tomogram_size': (960, 928)         # Tomogram dimensions (width, height)
    },
    
    # Add more tomograms here:
    # 'tomogram2': {
    #     'slab_range_start': 25,
    #     'slab_range_end': 175,
    #     'tomogram_size': (1024, 1024)
    # }
} 
