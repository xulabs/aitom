import tomominer.filter.gaussian as TFG
import tomominer.image.vol.wedge.util as TIVWU
from scipy.stats.stats import pearsonr
import tomominer.io.file as TIF
import numpy.fft as NF
import pickle
import uuid
import os
import sys
import gc
import multiprocessing
import itertools
import numpy as N
import tomominer.geometry.rotate as GR
import tomominer.geometry.ang_loc as AAL
import tomominer.simulation.reconstruction__eman2 as TSRE
import tomominer.image.vol.util as TIVU
import tomominer.image.io as TII
from tomominer.image.vol.util import cub_img
import tomominer.model.util as MU
import tomominer.align.util as TAU
import tomominer.parallel.multiprocessing.util as TPMU
reload(TPMU)


def simulation(SNR):
    img_size = 40

    dense_map_file = '/shared/u/xiangruz/for_CCQ/domain_rand/Qiang_4__density_maps.pickle'
    with open(dense_map_file) as f:
        m = pickle.load(f)
    #MCs = ['4R3O', '1KP8', '4V4A', '1BXR', '1AON', '1W6T']

    MCs = {'ribosome':300, 'membrane':300, 'TRiC':300, 'proteasome_s':300}
    #MCs = {"31": 400, "33": 400, "35": 400, "43": 400, "69": 400, "72": 400, "73": 400}
    subtomograms = {}
    for MC,num in MCs.items():
        v = m[MC]#[6.0][40.0]['map']
        subtomograms[MC] = []
        for i in range(num):
            loc_max = N.array(
                [img_size, img_size, img_size], dtype=float) * 0.4
            angle = AAL.random_rotation_angle_zyz()

            loc_r = (N.random.random(3)-0.5)*loc_max

            vr = GR.rotate(v, angle=angle, loc_r=loc_r, default_val=0.0)

            vrr = -TIVU.resize_center(vr, s=(img_size,
                                             img_size, img_size), cval=0.0)
            op = {'model': {'titlt_angle_step': 1, 'band_pass_filter': False, 'use_proj_mask': False, 'missing_wedge_angle': 30},
                  'ctf': {'pix_size': 1.368, 'Dz': -6.0, 'voltage': 300, 'Cs': 2.7, 'sigma': 0.4}}
            op['model']['SNR'] = SNR
            vb = TSRE.do_reconstruction(vrr, op, verbose=True)['vb']
            #TII.save_png(cub_img(vb)['im'], str(i)+'v.png')
            #subtomograms[MC].append({'v': vb, 'uuid': str(uuid.uuid4()), 'loc': loc_r, 'ang': angle, 'id': MC})
            subtomograms[MC].append(vb)


    return(subtomograms)


SNR = 0.01
subtomograms = simulation(SNR)
with open('../data/simulated1200.pickle','w') as f:
    pickle.dump(subtomograms, f)
