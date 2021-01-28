from . import simu_subtomo_single as SS
from .packing_single_sphere import pdb2ball_single as P2B
from .map_tomo import pdb2map as PM
from .map_tomo import iomap as IM
import os

op = {'map': {'situs_pdb2vol_program': '/shared/opt/local/img/em/et/util/situs/Situs_2.7.2/bin/pdb2vol',
              'spacing_s': [10.0], 'resolution_s': [10.0], 'pdb_dir': 'IOfile/pdbfile/',
              'out_file': '/IOfile/map_single/situs_maps.pickle', 'map_single_path': './IOfile/map_single'},
      # read density map from mrc
      'tomo': {'model': {'missing_wedge_angle': 30, 'SNR': 10},
               'ctf': {'pix_size': 1.0, 'Dz': -5.0, 'voltage': 300, 'Cs': 2.0, 'sigma': 0.4}},
      'target_size': 28,
      'v': None}


def mknewdir(s1):
    if not os.path.exists(s1):
        os.makedirs(s1)


mknewdir('./IOfile/pdbfile/')
mknewdir('./IOfile/map_single')
mknewdir('./IOfile/json/')
mknewdir('./IOfile/packmap/target/mrc')
mknewdir('./IOfile/packmap/target/png')
mknewdir('./IOfile/tomo/target/mrc')
mknewdir('./IOfile/tomo/target/png')

# convert pdb file into single ball and get the center and radius of this ball.
boundary_shpere = P2B.pdb2ball_single(PDB_ori_path='IOfile/pdbfile/', show_log=0)

packing_op = {  # 'target': '2byu',
    'random_protein_number': 0, 'PDB_ori_path': 'IOfile/pdbfile/', 'iteration': 5001, 'step': 1, 'show_img': 0,
    'show_log': 0, 'boundary_shpere': boundary_shpere}

# convert pdb to map
ms = PM.pdb2map(op['map'])
for n in ms:
    v = ms[n]
    IM.map2mrc(v, op['map']['map_single_path'] + '/{}.mrc'.format(n))
print('convert pdb to map')

# read density map from mrc
rootdir = op['map']['map_single_path']
v = IM.readMrcMapDir(rootdir)
print('read density map done')
op['v'] = v

# generate subtomogram dataset
num = 0
for num in range(1000):
    # '1w6t' '2bo9' '2gho' '3d2f' '1eqr' '1vrg' '3k7a' '1jgq' '1vpx'
    #  the above macromolecules may encounter some error

    # set target macromolecule name
    if num > 899:
        packing_op['target'] = '3hhb'
    elif num > 799:
        packing_op['target'] = '2h12'
    elif num > 699:
        packing_op['target'] = '2ldb'
    elif num > 599:
        packing_op['target'] = '6t3e'
    # else:
    #     print(num, 'Pass')
    elif num > 499:
        packing_op['target'] = '4d4r'
    elif num > 399:
        packing_op['target'] = '3gl1'
    elif num > 299:
        packing_op['target'] = '2byu'
    elif num > 199:
        packing_op['target'] = '1yg6'
    elif num > 99:
        packing_op['target'] = '1f1b'
    else:
        packing_op['target'] = '1bxn'

    if num >= 0:
        print(num, packing_op['target'], 'start')

        output = {'initmap': {'mrc': 'IOfile/initmap/mrc/initmap{}.mrc'.format(num),
                              'png': 'IOfile/initmap/png/initmap{}.png'.format(num),
                              'trim': 'Ofile/initmap/trim/initmap{}T.mrc'.format(num)},
                  'packmap': {'mrc': 'IOfile/packmap/mrc/packmap{}.mrc'.format(num),
                              'png': 'IOfile/packmap/png/packmap{}.png'.format(num),
                              'trim': 'IOfile/packmap/trim/packmap{}T.mrc'.format(num),
                              'target': {'mrc': 'IOfile/packmap/target/mrc/packtarget{}.mrc'.format(num),
                                         'png': 'IOfile/packmap/target/png/packtarget{}.png'.format(num)}},
                  'tomo': {'mrc': 'IOfile/tomo/mrc/tomo{}.mrc'.format(num),
                           'png': 'IOfile/tomo/png/tomo{}.png'.format(num),
                           'trim': 'IOfile/tomo/trim/tomo{}T.mrc'.format(num),
                           'target': {'mrc': 'IOfile/tomo/target/mrc/tomotarget{}.mrc'.format(num),
                                      'png': 'IOfile/tomo/target/png/tomotarget{}.png'.format(num)}},
                  'json': {'pack': 'IOfile/json/packing{}.json'.format(num),
                           'target': 'IOfile/json/target{}.json'.format(num)}}

        # do simulation
        SS.simu_subtomo(op, packing_op, output, save_tomo=0, save_target=1, save_tomo_slice=0)
        print(num, packing_op['target'], 'Done')
        num = num + 1

print('all Done')
