"""
a tutorial on using particle picking

Reference:
Pei et al. Simulating cryo electron tomograms of crowded cell cytoplasm for assessment of automated particle picking
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-1283-3
"""

import sys

from aitom.pick.dog.particle_picking_dog__util import peak__partition
from aitom.pick.dog.particle_picking_dog__filter import do_filter
import os
import json
import aitom.io.file as io_file
import aitom.image.vol.util as im_vol_util
from bisect import bisect
from pprint import pprint


def picking(path, s1, s2, t, find_maxima=True, partition_op=None,
            multiprocessing_process_num=0, pick_num=None):
    """
    Take a two-dimensional image as an example, if the image size is 210*150(all in pixels),
    nonoverlap_width is 60 and overlap_width is 30. It will be divided into 6 pieces
    for different threads to process. The ranges of their X and Y are
    (first line)  (0-90)*(0-90) (60-150)*(0-90) (120-210)*(0-90) (0-90)
    (second line) (0-90)*(60-150) (60-150)*(60-150) (120-210)*(60-150)
    In general, s2=1.1 * s1, s1 and t depend on particle size and noise.
    In practice, s1 should be roughly equal to the particle radius(in pixels).
    In related paper, the model achieves highest comprehensive score when s1=7 and t=3.

    parameters:
        path:file path
        s1:sigma1
        s2:sigma2
        t:threshold level
        find_maxima:peaks appears at the maximum/minimum
        multiprocessing_process_num: number of multiporcessing
        partition_op: partition the volume for multithreading,
            is a dict consists 'nonoverlap_width', 'overlap_width' and 'save_vg'
        pick_num: the max number of particles to pick out

    return:
        a list including all peaks information (in descending order of value),
        each element in the return list looks like:
        {'val': 281.4873046875, 'x': [1178, 1280, 0], 'uuid': '6ad66107-088c-471e-b65f-0b3b2fdc35b0'}
        'val' is the score of the peak when picking,
            only the score is higher than the threshold will the peak be selected.
        'x' is the center of the peak in the tomogram.
        'uuid' is an unique id for each peak.
    """
    a = io_file.read_mrc_data(path)
    print("file has been read")
    temp = im_vol_util.cub_img(a)
    # image data
    a_im = temp['im']
    # volume data
    a_vt = temp['vt']

    # using DoG to detect all peaks, may contain peaks caused by noise
    peaks = peak__partition(a_vt, s1=s1, s2=s2, find_maxima=find_maxima,
                            partition_op=partition_op,
                            multiprocessing_process_num=multiprocessing_process_num)
    '''
    calculate threshold T and delete peaks whose val are smaller than threshold
    Related paper: Pei L, Xu M, Frazier Z, Alber F. Simulating Cryo-Electron
            Tomograms of Crowded Mixtures of Macromolecular Complexes and Assessment
            of Particle Picking. BMC Bioinformatics. 2016; 17: 405.
    '''
    # max val of all peaks
    M = peaks[0]['val']
    # min val of all peaks
    m = peaks[len(peaks) - 1]['val']
    T = m + t * (M - m) / 20
    peak_vals_neg = [-peak['val'] * find_maxima for peak in peaks]
    res = peaks[:bisect(peak_vals_neg, -T * find_maxima) - 1]
    assert res[-1]['val'] >= T
    print("%d particles detected, containing redundant peaks" % len(res))
    # remove redundant peaks
    result = do_filter(pp=res, peak_dist_min=s1, op=None)
    print("peak number reduced to %d" % len(result))
    if pick_num is None:
        pass
    elif pick_num < len(res):
        res = res[:pick_num]

    print("T=m+t*(M-m)/20 \nT=%f m=%f t=%f M=%f" % (T, m, t, M))
    return res


def printUsage():
    print("Usage: This script will use aiTom picking to pick a tomogram.\n"
          "Invoke this file passing the PATH to a tomogram and the name of the output\n"
          "Example: particle_picking.py /tmp/mytomogram.mrc /tmp/coordinates3D.json")


def getParams():
    # Download from: https://cmu.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp
    if len(sys.argv) != 3:
        printUsage()
        raise AttributeError(
            "Wrong number of parameters. 2 expected, %s found: %s" %
            (len(sys.argv) - 1, sys.argv[1:]))

    path = sys.argv[1]

    if not os.path.exists(path):
        raise FileNotFoundError("File %s does not exists." % path)

    output = sys.argv[2]

    return path, output


def main():
    path, output = getParams()

    # Also, we can crop and only use part of the mrc image instead of binning for tasks requiring higher resolution
    # crop_path = 'cropped.mrc'
    # crop_mrc(path, crop_path)

    mrc_header = io_file.read_mrc_header(path)
    voxel_spacing_in_nm = mrc_header['MRC']['xlen'] / mrc_header['MRC']['nx'] / 10
    print("voxel_spacing_in_nm: %s" % voxel_spacing_in_nm)

    # Note: with our test data, voxel_spacing_in_nm has 0 and next division fails.
    sigma1 = 2
    try:
        # In general, 7 is optimal sigma1 val in nm according to the paper and sigma1 should at least be 2
        sigma1 = max(int(7 / voxel_spacing_in_nm), sigma1)
    except Exception as e:
        pass

    print('sigma1=%d' % sigma1)
    # For particular tomogram, larger sigma1 value may have better results.
    # Use IMOD to display selected peaks and determine best sigma1.
    # For 'aitom_demo_cellular_tomogram.mrc', sigma1 is 5 rather than 3 for better performance
    # (in this tomogram, 7nm corresponds to 3.84 pixels)
    # print(mrc_header['MRC']['xlen'], mrc_header['MRC']['nx'], voxel_spacing_in_nm, sigma1)

    partition_op = {'nonoverlap_width': sigma1 * 20,
                    'overlap_width': sigma1 * 10,
                    'save_vg': False}
    result = picking(path, s1=sigma1, s2=sigma1 * 1.1, t=3, find_maxima=False,
                     partition_op=partition_op, multiprocessing_process_num=100)
    print("%d particles detected, containing redundant peaks" % len(result))
    # remove redundant peaks
    result = do_filter(pp=result, peak_dist_min=sigma1, op=None)
    print("peak number reduced to %d" % len(result))
    pprint(result[:5])

    # generate file for 3dmod
    json_data = []
    for i in range(len(result)):
        loc_np = result[i]['x']
        loc = []
        for j in range(len(loc_np)):
            loc.append(loc_np[j].tolist())
        json_data.append({'peak': {'loc': loc}})

    with open(output, 'w') as f:
        json.dump(json_data, f)


if __name__ == '__main__':
    main()
