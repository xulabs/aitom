"""
The current whole tomogram simulator is developped according to the following paper, with some minor modifications.

Pei L, Xu M, Frazier Z, Alber F.
Simulating Cryo-Electron Tomograms of Crowded Mixtures of Macromolecular Complexes and Assessment of Particle Picking.
BMC Bioinformatics. 2016; 17: 405.

The example configuration and output files can be found at:
/shared/proj/190309-detection/tomogram-simulation/190331-0

The simulation pipeline is automated using following package
https://pypi.org/project/doit/


To generate data run:
doit

To clean up generated files run:
doit clean
rm -rf ./density_map ./tomogram

File contents

dodo.py: this file defines how the simulation modules are organized, and their file dependencies

density_map: this folder contains simulated density maps, in npy (numpy array), and mac format.
    The npy files can be directly loaded using numpy. The mrc file can be displayed using imod software for visual
    inspection.

tomogram: this folder contains simulated tomograms in npy format
density_map_to_tomogram__out_stat.json: information about which tomogram is generated from which density map,
corresponding SNR and missing wedge angle etc

template_select__op.json: this file can be used to select a subset of classes by changing:
    ['pdb_ids']

model_generation_imp__op.json: configuration file for packing macromolecules,
    ['copy_number']['total']: number of macromolecules to try to be packed into the volume,
        in most of cases, we want to have it to be large so that the volume can contain as many macromolecules as
        possible
        on the other hand, not too large to cost too much computation
        so as long as we see in the packing result (model_generation_imp__out.json), large number of macromolecules
        are outside the volume,
        this means the volume is too crowded to insert any additional macromolecules
    ['model_number']: number of packing models to be generated
    ['packing']['param']['box']: defines the size of the volume to pack the macromolecules


model_generation_imp__out.json: Output of geometrical packing for all models. It is a dictionary indexed by model id.
    For each model, 'instances' contain the spatial information of all macromolecules.
    'pdb_id' is the structural class of the macromolecule.
    'x': center of the minimal bounding sphere that encloses the macromolecule.
    'redius': redius of the minimal bounding sphere.
    'angle': orientation of the macromolecule.

back_projection_reconstruction__op.json: parameters for simulating tomograms from density maps.
    ['model']
        ['SNR_s']: a list of SNR levels to be simulated
        ['missing_wedge_angle_s']: a list of missing wedge angles, we normally set to 30
    ['ctf'] parameters for CTF function
        ['Dz']: defocus
        ['CS']: spherical abberivation



IMPORTANT: generating density maps and tomograms consume large amount of harddisk storage. Make sure to clean up
useless generated data to save space.
Also make sure to tune model number and volume size parameters to generate just the amount of data you need.
"""
