"""
if you want to run this file to generate only one subtomogram, please uncomment line 85-96 and comment line 97
if you want to run file generate_simu_subtomo to generate a lot of data, please comment line85-96 and uncomment line
97. This will save time.
"""

import numpy as np
import sys
import json

sys.path.append("..")

# set parameters for the simulation
num = 1

op = {'map': {'situs_pdb2vol_program': '/shared/opt/local/img/em/et/util/situs/Situs_2.7.2/bin/pdb2vol',
              'spacing_s': [10.0], 'resolution_s': [10.0], 'pdb_dir': 'IOfile/pdbfile/',
              'out_file': '/IOfile/map_single/situs_maps.pickle', 'map_single_path': './IOfile/map_single'},
      # read density map from mrc
      'tomo': {'model': {'missing_wedge_angle': 30, 'SNR': 500000000},
               'ctf': {'pix_size': 1.0, 'Dz': -5.0, 'voltage': 300, 'Cs': 2.0, 'sigma': 0.4}},
      'target_size': 32,
      'v': None}

packing_op = {'target': '1bxn',
              'random_protein_number': 4,
              'PDB_ori_path': op['map']['pdb_dir'],
              'iteration': 5001,
              'step': 1,
              'show_img': 1,
              'show_log': 1}

output = {'initmap': {'mrc': 'IOfile/initmap/mrc/initmap{}.mrc'.format(num),
                      'png': 'IOfile/initmap/png/initmap{}.png'.format(num),
                      'trim': 'Ofile/initmap/trim/initmap{}T.mrc'.format(num)},
          'packmap': {'mrc': 'IOfile/packmap/mrc/packmap{}.mrc'.format(num),
                      'png': 'IOfile/packmap/png/packmap{}.png'.format(num),
                      'trim': 'IOfile/packmap/trim/packmap{}T.mrc'.format(num),
                      'target': {'mrc': 'IOfile/packmap/target/mrc/packtarget{}.mrc'.format(num),
                                 'png': 'IOfile/packmap/target/png/packtarget{}.png'.format(num)}},
          'tomo': {'mrc': 'IOfile/tomo/mrc/tomo{}.mrc'.format(num), 'png': 'IOfile/tomo/png/tomo{}.png'.format(num),
                   'trim': 'IOfile/tomo/trim/tomo{}T.mrc'.format(num),
                   'target': {'mrc': 'IOfile/tomo/target/mrc/tomotarget{}.mrc'.format(num),
                              'png': 'IOfile/tomo/target/png/tomotarget{}.png'.format(num)}},
          'json': {'pack': 'IOfile/json/packing{}.json'.format(num), 'target': 'IOfile/json/target{}.json'.format(num)}}


def simu_subtomo(op, packing_op, output, save_tomo=0, save_target=1, save_tomo_slice=0):
    """
    :param
        op: parameter of convert pdb/ density map to tomogram
        packing_op: parameter of packing process
        output: path of output file
        save_tomo: 1 save, 0 not save, if the large subtomogram.mrc of the whole packing scene(including 5-10
            macromolecules) will be saved
        save_target: 1 save, 0 not save, if the subtomogram.mrc of the target will be saved
        save_tomo_slice: 1 save, 0 not save, if the sliced image will be saved.
    :return:
        target_simu_tomo = {'tomo': subtomogram of target macromolecule, .mrc file
                            'density_map': density map, .mrc file
                            'info': {'loc': coordinate of target macromolecule
                                    'rotate': the rotation angle of this macromolecule (ZYZ, Euler angle)
                                    'name': the name of a macromolecule}}

    the json file of packing result and target will be saved in '../IOfile/json'
    """
    from .map_tomo import pdb2map as PM
    from .map_tomo import iomap as IM
    from .map_tomo import mrc2singlepic as MS
    from .map_tomo import merge_map as MM
    from .map_tomo import map2tomogram as MT
    from .packing_single_sphere import simulate as SI

    # convert pdb to map
    # ms = PM.pdb2map(op['map'])
    # for n in ms:
    #     v = ms[n]
    #     IM.map2mrc(v, op['map']['map_single_path']+ '/{}.mrc'.format(n))

    # read density map from mrc
    # rootdir = op['map']['map_single_path']
    # v = IM.readMrcMapDir(rootdir)
    # print('read density map done')
    v = op['v']

    # get packing info
    target_name = packing_op['target']
    packing_result = SI.packing_with_target(packing_op)
    protein_name = packing_result['optimal_result']['pdb_id']
    x = packing_result['optimal_result']['x'] / 10
    y = packing_result['optimal_result']['y'] / 10
    z = packing_result['optimal_result']['z'] / 10
    # print('initialization',packing_result['optimal_result']['initialization'])
    x0 = np.array(packing_result['optimal_result']['initialization'][0]) / 10
    y0 = np.array(packing_result['optimal_result']['initialization'][1]) / 10
    z0 = np.array(packing_result['optimal_result']['initialization'][2]) / 10
    box_size = packing_result['general_info']['box_size'] / 10
    print('get packing info done')

    # merge map to hugemap, save random angle in packing_result
    initmap, init_angle_list = MM.merge_map(v, protein_name, x0, y0, z0, box_size)
    packmap, pack_angle_list = MM.merge_map(v, protein_name, x, y, z, box_size)
    packing_result['optimal_result']['initmap_rotate_angle'] = init_angle_list
    packing_result['optimal_result']['packmap_rotate_angle'] = pack_angle_list
    print('merge huge map done ')

    # save packing info
    with open(output['json']['pack'], 'w') as f:
        json.dump(packing_result, f, cls=MM.NumpyEncoder)
    print('save packing info')
    # convert packmap to tomogram
    tomo = MT.map2tomo(packmap, op['tomo'])
    print('convert to tomo')

    if save_tomo != 0:
        # save init & pack map to mrc
        IM.map2mrc(initmap, output['initmap']['mrc'])
        IM.map2mrc(packmap, output['packmap']['mrc'])
        # save init & pack map to png
        IM.map2png(initmap, output['initmap']['png'])
        IM.map2png(packmap, output['packmap']['png'])
        # save pack tomo to mrc and png
        IM.map2mrc(tomo, output['tomo']['mrc'])
        IM.map2png(tomo, output['tomo']['png'])

    # trim hugemap
    # trim_initmap = MM.trim_margin(initmap)
    # trim_packmap = MM.trim_margin(packmap)
    # trim_tomo = MM.trim_margin(tomo)
    # print('initmap shape',initmap.shape)
    # print('trimmed shape',trim_initmap.shape)
    # print('packmap shape',packmap.shape)
    # print('trimmed shape',trim_packmap.shape)
    # print('tomo shape',tomo.shape)
    # print('trimmed shape',trim_tomo.shape)
    # IM.map2mrc(trim_initmap, output['initmap']['trim'])
    # IM.map2mrc(trim_packmap, output['packmap']['trim'])
    # IM.map2mrc(trim_tomo, output['tomo']['trim'])

    # trim target
    i = protein_name.index(target_name)
    # print('i',i)
    target_packmap, loc_r = MM.trim_target(packmap, np.array([x[i], y[i], z[i]]), op['target_size'])
    target_tomo, loc_r = MM.trim_target(tomo, np.array([x[i], y[i], z[i]]), op['target_size'], loc_r)
    if save_target != 0:
        IM.map2mrc(target_packmap, output['packmap']['target']['mrc'])
        IM.map2mrc(target_tomo, output['tomo']['target']['mrc'])
        IM.map2png(target_packmap, output['packmap']['target']['png'])
        IM.map2png(target_tomo, output['tomo']['target']['png'])
    print('trim target done')

    # save target info
    target_info = dict()
    target_info['loc'] = loc_r
    target_info['rotate'] = pack_angle_list[i]
    target_info['name'] = packing_op['target']
    with open(output['json']['target'], 'w') as f:
        json.dump(target_info, f, cls=MM.NumpyEncoder)
    print('get target info done')

    if save_tomo_slice != 0:
        # convert packmap & tomo to separate pictures
        MS.mrc2singlepic(output['packmap']['mrc'], output['packmap']['png'] + 'packmap{}/'.format(num),
                         'packmap{}'.format(num))
        MS.mrc2singlepic(output['tomo']['mrc'], output['tomo']['png'] + 'tomo{}/'.format(num), 'tomo{}'.format(num))

    target_simu_tomo = dict()
    target_simu_tomo['tomo'] = target_tomo
    target_simu_tomo['density_map'] = target_packmap
    target_simu_tomo['info'] = target_info
    print('all done')

    return target_simu_tomo


if __name__ == '__main__':
    try:
        simu_subtomo(sys.argv[1], sys.argv[2], sys.argv[3], save_tomo=1, save_target=1, save_tomo_slice=1)
    except Exception as e:
        simu_subtomo(op, packing_op, output, save_tomo=0, save_target=1, save_tomo_slice=0)
