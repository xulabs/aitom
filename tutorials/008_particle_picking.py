'''
a tutorial on using particle picking

Reference:
Pei et al. Simulating cryo electron tomograms of crowded cell cytoplasm for assessment of automated particle picking
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-1283-3
'''


from aitom.pick.dog.particle_picking_dog__util import peak
from aitom.pick.dog.particle_picking_dog__util import peak__partition
import os
import json
os.chdir("..")
import aitom.io.file as io_file
import aitom.image.vol.util as im_vol_util
from aitom.filter.gaussian import smooth
from aitom.filter.gaussian import dog_smooth


'''
particle picking
parameters:  path: file path  s1:sigma1  s2:sigma2  t:threshold level  
In general, s2=1.1*s1, s1 and t depend on particle size and noise. In related paper, the model achieves highest comprehensive score when s1=7 and t=3. 

return:  a list including all peaks information, each element in the return list looks like: 
{'val': 281.4873046875, 'x': [1178, 1280, 0], 'uuid': '6ad66107-088c-471e-b65f-0b3b2fdc35b0'}

'''
def picking(path, s1, s2, t, find_maxima=True, partition_op=None, multiprocessing_process_num=0):

    a = io_file.read_mrc_data(path)
    print("file has been read")
    temp = im_vol_util.cub_img(a)
    a_im = temp['im'] # image data
    a_vt = temp['vt'] # volume data

    peaks = peak__partition(a_vt, s1=s1, s2=s2, find_maxima=find_maxima, partition_op=partition_op, multiprocessing_process_num=multiprocessing_process_num) 
    # using DoG to detect all peaks, may contain peaks caused by noise
    
    # calculate threshold T and delete peaks whose val are smaller than threshold
    # Related paper: Pei L, Xu M, Frazier Z, Alber F. Simulating Cryo-Electron Tomograms of Crowded Mixtures of Macromolecular Complexes and Assessment of Particle Picking. BMC Bioinformatics. 2016; 17: 405.

    M = peaks[0]['val'] # max val of all peaks
    m = peaks[len(peaks)-1]['val'] # min val of all peaks
    T = m+t*(M-m)/20
    for i in range(len(peaks)):
        if peaks[i]['val'] < T:
            res = peaks[0:i-1]
            break
    print("T=m+t*(M-m)/20 \nT=%f m=%f t=%f M=%f" %(T,m,t,M))
    return res
    
def main():
    # Download from: http://ftp.ebi.ac.uk/pub/databases/empiar/archive/10227/data/US1363_G1/14nov13a_tilt/14nov13a__001.mrc.bz2
    # Can first reduce its size through bining
    result=picking('/ldap_shared/home/v_zhenxi_zhu/14nov13a__001_bin2.mrc', s1=7, s2=7.7, t=3, find_maxima=True, partition_op=None, multiprocessing_process_num=100)
    print("%d particles deteced" %len(result))
    print(result[0])
    
if __name__ == '__main__':
    main()

