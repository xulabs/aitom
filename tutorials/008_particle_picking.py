'''
a tutorial on using particle picking

Reference:
Pei et al. Simulating cryo electron tomograms of crowded cell cytoplasm for assessment of automated particle picking
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-1283-3
'''


from aitom.pick.dog.particle_picking_dog__util import peak
from aitom.pick.dog.particle_picking_dog__util import peak__partition
from aitom.pick.dog.particle_picking_dog__filter import do_filter
import os
import json
os.chdir("..") # Depends on your current dir
import aitom.io.file as io_file
import aitom.image.vol.util as im_vol_util
from aitom.filter.gaussian import smooth
from aitom.filter.gaussian import dog_smooth
from bisect import bisect
from pprint import pprint
'''
parameters:
path:file path  s1:sigma1  s2:sigma2  t:threshold level  find_maxima:peaks appears at the maximum/minimum  multiprocessing_process_num: number of multiporcessing
partition_op: partition the volume for multithreading, is a dict consists 'nonoverlap_width', 'overlap_width' and 'save_vg'
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
def picking(path, s1, s2, t, find_maxima=True, partition_op=None, multiprocessing_process_num=0):
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
    peak_vals_neg = [-peak['val'] for peak in peaks]
    res = peaks[:bisect(peak_vals_neg, -T)]
    assert res[-1]['val'] >= T
    print("T=m+t*(M-m)/20 \nT=%f m=%f t=%f M=%f" %(T,m,t,M))
    return res
    
def main():
    # Download from: https://cmu.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp
    path = '/ldap_shared/home/v_zhenxi_zhu/data/aitom_demo_cellular_tomogram.mrc'
    
    mrc_header = io_file.read_mrc_header(path)
    voxel_spacing_in_nm = mrc_header['MRC']['xlen'] / mrc_header['MRC']['nx'] / 10
    sigma1 = max(int(7 / voxel_spacing_in_nm), 2) # 7 is optimal sigma1 val in nm according to the paper and sigma1 should at least be 2
    # print(mrc_header['MRC']['xlen'], mrc_header['MRC']['nx'], voxel_spacing_in_nm, sigma1)
    partition_op = {'nonoverlap_width': sigma1*20, 'overlap_width': sigma1*10, 'save_vg': False}
    result = picking(path, s1=sigma1, s2=sigma1*1.1, t=3, find_maxima=True, partition_op=partition_op, multiprocessing_process_num=100)
    print("%d particles detected, containing redundant peaks" % len(result))
    result = do_filter(pp=result, peak_dist_min=sigma1, op=None)  # remove redundant peaks
    print("peak number reduced to %d" % len(result))
    pprint(result[:5])
    
if __name__ == '__main__':
    main()

