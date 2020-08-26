#!/usr/bin/env python3

'''
arguments needed from job_file.p:
	mode
	template
    map_file
    job
    angles
    id
    out_dir
    stat_out
Demo data came from: https://cellix.imba.oeaw.ac.at/lamellipodia/et

To run this script:
python3 <filepath>/template_matching_demo.py <filepath>/job_file.py

output: c.npy, phi.npy, theta.npy, psi.npy
'''

import os, sys, json, time
import pickle
import numpy as N
import scipy.ndimage.filters as SNF

import aitom.tomominer.io.file as IOF
import aitom.geometry.rotate as GR
import aitom.filter.convolve as FC #convolution
import aitom.filter.normalized_cross_correlation as FNCC #cross correlation
import aitom.filter.local_extrema as FLE

import mrcfile

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

            c = FNCC.cor(v=v, t=tr) #calculate Pearson cross correlation
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
    #load input values from job_file.p
    job_file = sys.argv[1]
    print ('loading job from', job_file)

    with open(job_file, 'rb') as f:     jop = pickle.load(f)
    print ('to process', len(jop['angles']), 'rotation angles')        ;       sys.stdout.flush()

    #parse input parameters, see generate_job_file.py
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