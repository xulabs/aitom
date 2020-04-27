# Simple-Packing
a simple packing function that implements 'single ball model' and 'multiple-ball model' for packing only 5 - 10 macromolecules. So that we can simulate a subtomogram that contain one target macromolecule and some neighbor structures. And the simulation can be performed quickly in CPU.


# Workflow of single-ball-model packing
run simulate.py, a packing result with a default parameter will be obtained
 
    
###### 1.pbd 2 single ball
    
###### 2.set target protein

###### 3.random select neighbor proteins

###### 4.set simulate box size

###### 5.initialization the location of all proteins

###### 6. packing using gradient descent

###### 7. run multiple times and return the optimal packing result
    
###### 8. return all important information
