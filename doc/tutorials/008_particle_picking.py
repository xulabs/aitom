'''
a tutorial on using particle picking

Reference:
Pei et al. Simulating cryo electron tomograms of crowded cell cytoplasm for assessment of automated particle picking
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-1283-3
'''


from aitom.pick.dog.particle_picking_dog__util import peak
from aitom.pick.dog.particle_picking_dog__util import peak__partition
from aitom.pick.dog.particle_picking_dog__filter import do_filter
from aitom.io.mrc.crop import crop_mrc
from aitom.pick.plot.particle_location_display_imod import generate_lines
from aitom.pick.plot.particle_location_display_imod import display_map_with_lines
import os
import json
import numpy as N
os.chdir("..") # Depends on your current dir
import aitom.io.file as io_file
import aitom.image.vol.util as im_vol_util
from aitom.filter.gaussian import smooth
from aitom.filter.gaussian import dog_smooth
from bisect import bisect
from pprint import pprint
import aitom.io.mrcfile_proxy as TIM

def picking(path, s1, s2, t, find_maxima=True, partition_op=None, multiprocessing_process_num=0, pick_num=None):
    '''
    parameters:
    path:file path  s1:sigma1  s2:sigma2  t:threshold level  find_maxima:peaks appears at the maximum/minimum  multiprocessing_process_num: number of multiporcessing
    partition_op: partition the volume for multithreading, is a dict consists 'nonoverlap_width', 'overlap_width' and 'save_vg'
    pick_num: the max number of particles to pick out
    # Take a two-dimensional image as an example, if the image size is 210*150(all in pixels), nonoverlap_width is 60 and overlap_width is 30.
    # It will be divided into 6 pieces for different threads to process. The ranges of their X and Y are
    # (first line)  (0-90)*(0-90) (60-150)*(0-90) (120-210)*(0-90) (0-90)
    # (second line) (0-90)*(60-150) (60-150)*(60-150) (120-210)*(60-150)
    In general, s2=1.1*s1, s1 and t depend on particle size and noise. In practice, s1 should be roughly equal to the particle radius(in pixels). In related paper, the model achieves highest comprehensive score when s1=7 and t=3. 

    return:
    a list including all peaks information (in descending order of value),  each element in the return list looks like: 
    {'val': 281.4873046875, 'x': [1178, 1280, 0], 'uuid': '6ad66107-088c-471e-b65f-0b3b2fdc35b0'}
    'val' is the score of the peak when picking, only the score is higher than the threshold will the peak be selected.
    'x' is the center of the peak in the tomogram.
    'uuid' is an unique id for each peak.
    '''
    a = io_file.read_mrc_data(path)
    print("file has been read")
    temp = im_vol_util.cub_img(a)
    a_im = temp['im'] # image data
    a_vt = temp['vt'] # volume data

    # using DoG to detect all peaks, may contain peaks caused by noise
    peaks = peak__partition(a_vt, s1=s1, s2=s2, find_maxima=find_maxima, partition_op=partition_op, multiprocessing_process_num=multiprocessing_process_num) 
    
    # calculate threshold T and delete peaks whose val are smaller than threshold
    # Related paper: Pei L, Xu M, Frazier Z, Alber F. Simulating Cryo-Electron Tomograms of Crowded Mixtures of Macromolecular Complexes and Assessment of Particle Picking. BMC Bioinformatics. 2016; 17: 405.
    M = peaks[0]['val'] # max val of all peaks
    m = peaks[len(peaks)-1]['val'] # min val of all peaks
    T = m+t*(M-m)/20
    peak_vals_neg = [-peak['val']*find_maxima for peak in peaks]
    res = peaks[:bisect(peak_vals_neg, -T*find_maxima)-1]
    assert res[-1]['val'] >= T
    print("%d particles detected, containing redundant peaks" % len(res))
    result = do_filter(pp=res, peak_dist_min=s1, op=None)  # remove redundant peaks
    print("peak number reduced to %d" % len(result))
    if pick_num is None:
        pass
    elif pick_num < len(res):
        res = res[:pick_num]

    print("T=m+t*(M-m)/20 \nT=%f m=%f t=%f M=%f" %(T,m,t,M))
    return res
    
def main():
    # Download from: https://cmu.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp
    path = '/ldap_shared/home/v_zhenxi_zhu/data/aitom_demo_single_particle_tomogram.mrc'
    
    # Also, we can crop and only use part of the mrc image instead of binning for tasks requiring higher resolution
    # crop_path = 'cropped.mrc'
    # crop_mrc(path, crop_path)
    
    mrc_header = io_file.read_mrc_header(path)
    voxel_spacing_in_nm = mrc_header['MRC']['xlen'] / mrc_header['MRC']['nx'] / 10
    sigma1 = max(int(7 / voxel_spacing_in_nm), 2) # In general, 7 is optimal sigma1 val in nm according to the paper and sigma1 should at least be 2
    print('sigma1=%d' %sigma1)
    # For particular tomogram, larger sigma1 value may have better results. Use IMOD to display selected peaks and determine best sigma1.
    # For 'aitom_demo_cellular_tomogram.mrc', sigma1 is 5 rather than 3 for better performance(in this tomogram, 7nm corresponds to 3.84 pixels)
    # print(mrc_header['MRC']['xlen'], mrc_header['MRC']['nx'], voxel_spacing_in_nm, sigma1)
    
    partition_op = {'nonoverlap_width': sigma1*20, 'overlap_width': sigma1*10, 'save_vg': False}
    result = picking(path, s1=sigma1, s2=sigma1*1.1, t=3, find_maxima=False, partition_op=partition_op, multiprocessing_process_num=10, pick_num=1000)
    print("DoG done, %d particles picked" % len(result))
    pprint(result[:5])
    
    # (Optional) Save subvolumes of peaks for autoencoder input
    dump_subvols = True
    if dump_subvols: # use later for autoencoder
        subvols_loc = "demo_single_particle_subvolumes.pickle"
        from aitom.classify.deep.unsupervised.autoencoder.autoencoder_util import peaks_to_subvolumes
        a = io_file.read_mrc_data(path)
        d = peaks_to_subvolumes(im_vol_util.cub_img(a)['vt'], result, 32)
        io_file.pickle_dump(d, subvols_loc)
        print("Save subvolumes .pickle file to:", subvols_loc)
        
    # Display selected peaks using imod/3dmod (http://bio3d.colorado.edu/imod/)
    '''
    #Optional: smooth original image
    a = io_file.read_mrc_data(path) 
    path =path[:-5]+'_smoothed'+path[-4:]
    temp = im_vol_util.cub_img(a)
    s1 = sigma1
    s2=sigma1*1.1
    vg = dog_smooth(temp['vt'], s1,s2)
    #vg = smooth(temp['vt'], s1)
    TIM.write_data(vg,path)
    '''  
    json_data=[] # generate file for 3dmod
    for i in range(len(result)):
        loc_np=result[i]['x']
        loc=[]
        for j in range(len(loc_np)):
            loc.append(loc_np[j].tolist())    
        json_data.append({'peak':{'loc':loc}}) 
    with open('data_json_file.json','w') as f:
        json.dump(json_data,f)

    dj=json_data
    x = N.zeros(    (len(dj), 3)  )
    for i,d in enumerate(dj):        x[i,:] = N.array(d['peak']['loc'])

    l = generate_lines(x_full=x, rad=sigma1)
    display_map_with_lines(l=l, map_file=path)
    
    
if __name__ == '__main__':
    main()

