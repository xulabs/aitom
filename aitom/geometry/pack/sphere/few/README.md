# Simple-Packing
a simple packing function that implements 'single ball model' and 'multiple-ball model' for packing only 5 - 10 macromolecules. So that we can simulate a subtomogram that contain one target macromolecule and some neighbor structures. And the simulation can be performed quickly in CPU.


# including the following algorithm
###### 1. packing_single_sphere: using single sphere to model one macromolecule, packing using gradient descent.

###### 2. map_tomo: obtain density map of single macromolecule and merge them based on packing result, then convert to cryo-ET.

###### 3. simu_subtomo.py: packing and obtain cryo-ET of several macromolecules, then get the subtomogram of the target macromolecule.

###### 4. generate_simu_subtomo.py: generate a set of simulated subtomogram automatically.