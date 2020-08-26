#!/usr/bin/env python3
'''
Script to generate an example job_file.p as the input for template_matching_demo.py
id - id of the current job
out_dir - output directory
template - filepath of the template (a mrc or rec file)
mode - "convolve" or "normalized-cor" 
angles - a list of (phi, theta, psi) tuples in the ZYZ convention for the template to be rotated (see Euler angles)
map_file - filepath to the subtomogram to be analized
stat_out - name or filepath of the output file generated with pickle.dump()

increment - the angular increment (in radian) for the template to be rotated along the axes in the ZYZ convention
Note that the code below can be easily modified to accommodate different increment values for phi, theta, and psi.

To run this script:
python3 <filepath>/generate_job_file.py
'''


import math
import pickle

#demo increment = 30 degree
#360/30 = 12 rotations along each of the 3 axes
#12^3 = 1728 template rotations in total
#run time ~70min
increment = (1/6)*math.pi #TODO: adjust increment for more accuracy or better run time
print("angle increment: " , round(180.0*increment/math.pi), "degree" )
print(increment)


def generate_all_rotation_angles(increment):
	'''
	generates all the ZYZ rotation angle combinations for the template from a user-specified increment value
	n^3 combinations in total for z, y, z 
	e.g. (4.71238898038469, 4.1887902047863905, 1.5707963267948966) for (phi, theta, psi)
	returns a python array of angle 3-tuples
	'''
	n = round( (2*math.pi) / increment) #number of rotations along one axis, assume int (e.g. 0, 45, 90, 135, 180, 225, 270, 315,  8 rotations, in radian)
	print(n, "rotations along each axis")
	print("generating template rotations...")

	angles = [] # a python array of angle tuples
	phi = 0.0
	theta = 0.0
	psi = 0.0
	for i in range(n): #phi, rotation about Z
		phi = i * increment
		for j in range(n): #theta, rotation about Y
			theta = j * increment
			for k in range(n): #psi, rorarion about Z
				psi = k * increment
				angles.append((phi, theta, psi))

	return angles

angles = generate_all_rotation_angles(increment) #generate all angle combinations for template rotation
print(len(angles) ," angle combinations in total for template rotation")

#TODO: change file paths
job_file = {	"id": "test_id",
				"out_dir":"/Users/xueyaoguo/Desktop/template_matching_tutorial/results", #TODO
				"template":"/Users/xueyaoguo/Desktop/template_matching_tutorial/demo_template.rec", #TODO
				"mode":"normalized-cor", 
				"angles": angles,
				"map_file":"/Users/xueyaoguo/Desktop/template_matching_tutorial/demo_map.rec",#TODO
				"stat_out":"/Users/xueyaoguo/Desktop/template_matching_tutorial/results/out.p" #TODO
			}

pickle.dump(job_file, open("/Users/xueyaoguo/Desktop/template_matching_tutorial/job_file.p", "wb")) #TODO
print("job_file.p generated")