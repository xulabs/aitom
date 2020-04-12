
# Readme
The main file to be run is named 'simulate.py' with no parameters. To change some settings such as temperature, mixture content, etc. will require you to go into this file and make some modifications (sorry I didn't think this through when I coded this, I'll change this in the future)
I can walk you through the file content in the following sections

# Python Files
Here's the brief into to the several python files in the folder


## simulate.py

The main runner of the project. It is dependent on pack.py and boundingsphere.py 
Right now it contains a dictionary of 20 proteins I used in my project with calculated bounding sphere and mass so you don't have to compute them every time. However, there are functions inside this file to calculate both bounding sphere and mass if the protein is not found. 

Function find_bounding_sphere will give return the center and the radius of the bounding sphere given a path to the mrc file and a countour level L. 

Function get_mass will return a 'relative mass' of a protein given the pdb file. Note that this mass is just the atom count from the pdb file, it is not the actual mass stored in the dictionary. **If you decide to use this function to compute the mass of your protein, make sure that the mass of all other proteins used are also computed this way so the relative mass stays the same.** I.e. don't use half of the protein in the stored dictionary and half calculated using this function

Finally the get_distribution will return a protein mixture using multinomial distribution of random frequency. The function takes in a list of strings of protein names that match the key in a dictionary that stores the information of said proteins. Note that this function does not check for duplicates. **This function requires a folder names pdb/ and a folder named mrc/ that stores the pdb and mrc files respectively. The file names must be id.pdb or id.mrc where id is the exact match of the strings passed to this function**

The call to do_packing does the actual simulation. directly pass the protein mixture returned from get_distribution without modification should be fine. The parameters of this call is something you might want to change to suit your project. T**he 'box' entry in the paramters defines the size of the container in which the mixture of the proteins will reside. Temperature is the initial temperature for the annealing process. Temperature decrease is the amount you decrease if the score stays the same for sometime. step is the optimization step to take. Recent score is how long do you decrease if you don't see score decrease. recent min score is the threshold of score decrease before you decrease the temperature.** pymol_file_name is optional

Once the packing process is complete, it stores the protain final location in a file called 

 - 'particles.npy' file. It's an array of proteins. 
 - proteins[i] stores a dictionary of a protein  
 - proteins[i]['x'] is the final location of the bounding sphere of the protein
     - proteins[i]['x'][0] is the x position
     - proteins[i]['x'][1] is the y position 
     - proteins[i]['x'][2] is the z position
 - proteins[i]['r'] is the radius of the protein
 - proteins[i]['mass'] is the mass
 - proteins[i]['id'] is the name of the protein
 - proteins[i]['c'] is the color that I put for graphing, this is quite irrelevant, don't worry about this

The final part of the file is graphing. 

## boundingSphere.py

calculates the bounding sphere of a set of points, sometimes gives error (I have yet to debug this file). Strongly recommend using the boundingsphere function in the simulate.py file which uses a package called miniball to calculate bounding sphere, much more reliable. 

## pack.py

this is the file that does the actual packing. Requires IMP module. 

The only thing you might want to change is the variable expand_n. This defines the initial starting space of the proteins. For example, if your container is of size 60, 60, 60, and your expand_n is 2, then  your proteins will be placed at a random position from 0, 0, 0, to 120, 120, 120 and slowly packed into the box. In my experience, having expand_n = 1 will have overlaps but having expand_n > 1 will usually result in long running time. Use your own discretion 

# Requirements

The following packages are used in this project. 

## Biopython

[https://biopython.org/wiki/Download](https://biopython.org/wiki/Download) 

This now supports python3, but I've experienced some issues in the past. 

## mrcfile 
`pip install mrcfile` or 
`python3 -m pip install mrcfile`  if pip not available 

[https://mrcfile.readthedocs.io/en/latest/usage_guide.html](https://mrcfile.readthedocs.io/en/latest/usage_guide.html) 

## mayavi

for graphing spheres. 

[https://docs.enthought.com/mayavi/mayavi/installation.html](https://docs.enthought.com/mayavi/mayavi/installation.html) 


## miniball

for finding the minimum boundingsphere 

`pip install miniball` or 

    python3 -m pip install miniball 
   if pip is not available. 

# Reference

Pei L, Xu M, Frazier Z, Alber F. Simulating Cryo-Electron Tomograms of Crowded Mixtures of Macromolecular Complexes and Assessment of Particle Picking. BMC Bioinformatics. 2016; 17: 405.
