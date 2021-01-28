# Line 22 is the function to calculate the minimum boundary sphere which is very useful in packing,
# please change the PDB_ori_path in line 25 to the folder of input PDB files.
# If you already have the density map to each single macromolecule, comment line 40-44,
# then change the op['map']['map_single_path'] to the path of the single density map.
# Change the input value in line 53 to the number of targets and modify the PDB id in line 53-62.
# Change the number in line 67 to the number of subtomograms for each target.
# change the save path in line 71-82 to the folder you want.

import simu_map as SS
from packing_single_sphere import pdb2ball_single as P2B

op = {
    'map':{'situs_pdb2vol_program':'/shared/opt/local/img/em/et/util/situs/Situs_2.7.2/bin/pdb2vol',
           'spacing_s': [10.0], 'resolution_s':[10.0],
           'pdb_dir':'IOfile/pdbfile/',
           'out_file':'/IOfile/map_single/situs_maps.pickle',
           'map_single_path': './IOfile/map_single'}, # read density map from mrc
    'tomo':{'model':{'missing_wedge_angle':30, 'SNR':0.3},
            'ctf':{'pix_size':1.0, 'Dz':-5.0, 'voltage':300, 'Cs':2.0, 'sigma':0.4}},
    'target_size':32,
    'v': None
    }

# convert pdb file into single ball and get the center and radius of this ball.
boundary_shpere = P2B.pdb2ball_single(PDB_ori_path = 'IOfile/pdbfile/', show_log = 0)

packing_op = {#'target': '2byu',
              'random_protein_number': 4, #this is the neighbor number in the simulation sence. If you only want a single macromolecule, set this as zero.
              'PDB_ori_path': 'IOfile/pdbfile/',
              'iteration':5001,
              'step':1,
              'show_img': 0,
              'show_log': 0,
              'boundary_shpere':  boundary_shpere
              }

# convert pdb to map
import map_tomo.pdb2map as PM
import map_tomo.iomap as IM
ms = PM.pdb2map(op['map'])
for n in ms:
    v = ms[n]
    IM.map2mrc(v, op['map']['map_single_path']+ '/{}.mrc'.format(n))
print('convert pdb to map')

# read density map from mrc
rootdir = op['map']['map_single_path']
v = IM.readMrcMapDir(rootdir)
print('read density map done')
op['v'] = v

# generate simulated data
for pdb_index in range(1):
    if pdb_index == 0:
        packing_op['target'] = '1bxn'
    elif pdb_index == 1:
        packing_op['target'] = '1f1b'
    elif pdb_index == 2:
        packing_op['target'] = '1yg6'
    elif pdb_index == 3:
        packing_op['target'] = '2byu'
    else:
        packing_op['target'] = '4d4r'

    # generate a set of subtomogram dataset for one target
    num = 0
    for num in range(1):
        pdbid = packing_op['target']
        print(pdb_index, num, packing_op['target'])
        # set save path
        output = {
            'packmap': {
                'target': {
                    'mrc': 'IOfile/test/' + pdbid + '/map{}.mrc'.format(num),
                    'png': 'IOfile/test/' + pdbid + '/map{}.png'.format(num)}},
            'tomo': {
                'target': {
                    'mrc': 'IOfile/test/' + pdbid + '/tomo{}.mrc'.format(num),
                    'png': 'IOfile/test/' + pdbid + '/tomo{}.png'.format(num)}},
            'json': {
                'pack': 'IOfile/test/' + pdbid + '/packing{}.json'.format(num),
                'target': 'IOfile/test/' + pdbid + '/target{}.json'.format(num)}}

        # do simulation
        SS.simu_subtomo(op, packing_op, output, save_tomo=0, save_target=1, save_tomo_slice=0)
        print(pdb_index, num, packing_op['target'], 'Done')
        num = num + 1

print('all Done')
