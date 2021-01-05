import pickle, uuid, os, sys, gc, multiprocessing, itertools                                                                                                                                                
import numpy as N                                                                                                                                                                                           
import tomominer.geometry.rotate as GR                                                                                                                                                                      
import tomominer.geometry.ang_loc as AAL                                                                                                                                                                    
import tomominer.simulation.reconstruction__eman2 as TSRE
# import tomominer.simulation.reconstruction__simple_convolution as TSRSC
import tomominer.image.vol.util as TIVU                                                                                                                                                                     
import tomominer.io.file as TIF                                                                                                                                                                             

                                                                                                                                                                                                            
def simulation(num, SNR, missing_wedge_angle, MCs, dense_map_file):                                                                                                                
    img_size = 32                                                                                                                                                                                  
                                                                                                                                                                                                            
    with open(dense_map_file) as f:     m = pickle.load(f)                                                                                                                                                  
    subtomograms = []                                                                                                                                                                                       
    for MC in MCs:                                                                                                                                                                                      
        for i in range(num):                                                                                                                                                                                
            v = m[MC][12.0][12.0]['map']                                                                                                                                                           
            loc_max = N.array([img_size,img_size,img_size], dtype=float) * 0.4                                                                                                                              
            angle = AAL.random_rotation_angle_zyz()                                                                                                                                                     
                                                                                                                                                                                                            
            loc_r = N.round((N.random.random(3)-0.5)*loc_max)                                                                                                                                           
                                                                                                                                                                                                            
            vr = GR.rotate(v, angle=angle, loc_r = loc_r, default_val=0.0)                                                                                                                              
                                                                                                                                                                                                            
            vrr = -TIVU.resize_center(vr, s=(img_size, img_size, img_size), cval=0.0)                                                                                                              
            op = {'model':{'titlt_angle_step':1, 'band_pass_filter':False, 'use_proj_mask':False},                                                                               
            'ctf':{'pix_size':1.2, 'Dz':-2.0, 'voltage':300, 'Cs':2.7, 'sigma':0.4}}                                                                                                               
            op['model']['SNR'] = SNR
            op['model']['missing_wedge_angle'] = missing_wedge_angle                                                                                                                        
            vb = TSRE.do_reconstruction(vrr, op, verbose=True)                                                                                                                                         
            subtomograms.append(vb)                                                                                                                                                                     
                                                                                                                                                                                                            
    return(subtomograms)                                                                                                                                                                                


if __name__ == '__main__':
    dense_map_file = 'situs_pdb2vol__batch__out_120_120.pickle'
    
    b = {}
    b['5T2C_data'] = simulation(100, 0.1, 30, ['5T2C'], dense_map_file)
    b['1KP8_data'] = simulation(100, 0.1, 30, ['1KP8'], dense_map_file)                                                                     
    b['5T2C_template'] = simulation(1, N.float('Inf'), 0, ['5T2C'], dense_map_file)[0]
    b['1KP8_template'] = simulation(1, N.float('Inf'), 0, ['1KP8'], dense_map_file)[0]
                                                                     
    TIF.pickle_dump(b, 'aitom_demo_subtomograms.pickle')                                                                                      
                                                                                                                                                                                                            
                                                                                                                                                                                                        

