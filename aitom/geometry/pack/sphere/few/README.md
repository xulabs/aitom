# Simple-Packing
a simple packing function that implements 'single ball model' and 'multiple-ball model' for packing only 5 - 10 macromolecules. So that we can simulate a subtomogram that contain one target macromolecule and some neighbor structures. And the simulation can be performed quickly in CPU.


# including the following algorithm
###### 1. packing_single_sphere: using single sphere to model one macromolecule, packing using gradient descent.

###### 2. map_tomo: obtain density map of single macromolecule and merge them based on packing result, then convert to cryo-ET.

# how to use
###### 1. activate python2 environment and run generate_map.py. Need to prepare pdb files as input. Need to modify the I/O path before running.

###### 2. activate python3 environment and run generate_tomo.py. Need to use gpu0/gpu1 server and import aitom_core package.

# reference
###### [1] Liu S, Ma Y, Ban X, Zeng X, Nallapareddy V, Chaudhari A, Xu M. Efficient Cryo-Electron Tomogram Simulation of Macromolecular Crowding with Application to SARS-CoV-2[C]. 2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE. 2020.