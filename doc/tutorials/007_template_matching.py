#!/usr/bin/env python3
'''
Template matching pipeline
Demo data from: https://cellix.imba.oeaw.ac.at/lamellipodia/et
To run this script:
python3 <filepath>/template_matching_pipeline.py
Output: c.npy, phi.npy, theta.npy, psi.npy
'''

import os, sys, json, time, math
import pickle
import numpy as N
import scipy.ndimage.filters as SNF

import aitom.tomominer.io.file as IOF
import aitom.geometry.rotate as GR
import aitom.filter.convolve as FC #convolution
import aitom.filter.normalized_cross_correlation as FNCC #cross correlation
import aitom.filter.local_extrema as FLE

import mrcfile

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

def scan(op):
    '''
    This function contains:
        template and map preprocessing
        template rotations using ZYZ Euler angles
        cross-correlation calculation
    output: c.npy, phi.npy, theta.npy, psi.npy
    '''

    if not os.path.isdir(op['out_dir']):        os.makedirs(op['out_dir'])

    re = {} #create output file paths
    re['c'] = os.path.join(op['out_dir'], '%s-c.npy'%(op['id'], ))
    re['phi'] = os.path.join(op['out_dir'], '%s-phi.npy'%(op['id'], ))
    re['theta'] = os.path.join(op['out_dir'], '%s-theta.npy'%(op['id'], ))
    re['psi'] = os.path.join(op['out_dir'], '%s-psi.npy'%(op['id'], ))

    #if the output files already exits, template matching would not run
    if os.path.isfile(re['c']) and os.path.isfile(re['phi']) and os.path.isfile(re['theta']) and os.path.isfile(re['psi']):     return re

    #load template as t and preprocess
    mrc = mrcfile.open(op['template'], mode='r+',permissive=True)
    t = mrc.data
    print('template size',t.shape)
    tm = N.isfinite(t)      # real space mask
    t_mean = t[tm].mean()
    t[N.logical_not(tm)] = t_mean 
    tm = tm.astype(N.float)

    #load map as v
    mrc = mrcfile.open(op['map'], mode='r+',permissive=True)
    v = mrc.data
    v = v.astype(N.float)

    print ('map size', v.shape)           ;       sys.stdout.flush()

    diff_time_v = [] 
    cur_time = time.time()

    c_max = None
    phi_max = None
    theta_max = None
    psi_max = None

    #template rotations
    for i, (phi, theta, psi) in enumerate(op['angles']): 
        #print(i, (phi, theta, psi))
    
        #template rotation with affine transformation
        tr = GR.rotate(t, angle=(phi, theta, psi), default_val=t_mean) 
        tr = tr.astype(N.float)
        
        if op['mode'] == 'convolve': #convolution
            #c = FC.convolve(v=v, t=tr)
            c = SNF.convolve(input=v, weights=tr, mode='reflect')
        elif op['mode'] == 'normalized-cor': #correlation
            tmr = GR.rotate(tm, angle=(phi, theta, psi), default_val=0.0)
            tr[tmr < 0.5] = float('NaN')

            c = FC.pearson_correlation_simple(v=v, t=tr) #calculate Pearson cross correlation
        else:
            raise Exception('mode')

        #record the maximum cross-correlation coefficient and the orientation
        #c_max - a 3d numpy array of maximum cross-correlation coefficients
        #phi_max, theta_max, psi_max - 3d numpy arrays that store the maximum phi, theta, psi values (template orientation in ZYZ convention)
        if c_max is None:
            c_max = c
            phi_max = N.zeros(c.shape) + phi
            theta_max = N.zeros(c.shape) + theta
            psi_max = N.zeros(c.shape) + psi

        else:

            ind = (c > c_max)
            c_max[ind] = c[ind]
            phi_max[ind] = phi
            theta_max[ind] = theta
            psi_max[ind] = psi

        #run time
        diff_time = time.time()-cur_time
        diff_time_v.append(diff_time)
        remain_time = N.array(diff_time_v).mean() * (len(op['angles']) - i - 1)
        print ('angle', i+1, 'of', len(op['angles']), '; time used', diff_time, '; time remain', remain_time)                 ;           sys.stdout.flush()

        cur_time = time.time()

    #output
    if not os.path.isfile(re['c']):    
        N.save(re['c'], c_max)
        N.save(re['phi'], phi_max)
        N.save(re['theta'], theta_max)
        N.save(re['psi'], psi_max)

    return re

    if os.path.isfile(re['c']) and os.path.isfile(re['phi']) and os.path.isfile(re['theta']) and os.path.isfile(re['psi']):     return re

def main(): 
    '''
    demo increment = 30 degree
    360/30 = 12 rotations along each of the 3 axes
    12^3 = 1728 template rotations in total
    run time ~70min
    '''
    increment = (1/6)*math.pi #TODO: adjust increment for more accuracy or better run time
    print("angle increment: " , round(180.0*increment/math.pi), "degree" )
	
    angles = generate_all_rotation_angles(increment) #generate all angle combinations for template rotation
    print(len(angles) ," angle combinations in total for template rotation")

    #input parameters
    #TODO: change file paths
    jop = {	"id": "test_id", #TODO: id of the current job
            "out_dir":"./template_matching_tutorial/results", #TODO: output directory
            "template":"./template_matching_tutorial/demo_template.rec", #TODO: filepath of the template (a mrc or rec file)
            "mode":"normalized-cor", #"convolve" or "normalized-cor" 
            "angles": angles, #a list of (phi, theta, psi) tuples in the ZYZ convention for the template to be rotated (see Euler angles)
            "map_file":"./template_matching_tutorial/demo_map.rec",#TODO: filepath to the subtomogram to be analized
            "stat_out":"./template_matching_tutorial/results/out.p" #TODO: name or filepath of the output file generated with pickle.dump()
            }

    print ('to process', len(jop['angles']), 'rotation angles')        ;       sys.stdout.flush()

    #parse input parameters 
    sop = {'id': jop['id']}
    sop['out_dir'] = jop['out_dir']
    sop['template'] = jop['template']
    sop['mode'] = jop['mode']
    sop['angles'] = jop['angles']
    sop['map'] = jop['map_file']

    #template scanning
    s = scan(sop)

    #output to out.p
    with open(jop['stat_out'], 'w') as f:        json.dump(s, f, indent=2)
    print ('program finished')


if __name__ == "__main__":
    main()
