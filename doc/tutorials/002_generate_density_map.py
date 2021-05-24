"""
This script converts pdb files to density maps
"""


'''
op contains parameters for the pdb2vol probgram:

situs_pdb2vol_program: the location of pdb2vol program
pdb2vol can be found under one of the following locations. Set situs_pdb2vol_program accordingly
/shared/opt/local/img/em/et/util/situs/Situs_2.7.2/bin/pdb2vol
/opt/local/img/em/et/util/situs/Situs_2.7.2/bin/pdb2vol

spacing_s: is the voxel spacing in Anstron
resolution_s: the image resolution in Anstron
pdb_dir: the directory that contains pdb files. Could be your own directory or /shared/shared/data/pdb or /shared/data/pdb
out_file: the output file that contains converted density maps
'''
op = {'situs_pdb2vol_program': '/shared/opt/local/img/em/et/util/situs/Situs_2.7.2/bin/pdb2vol',
      'spacing_s': [10.0],
      'resolution_s': [10.0],
      'pdb_dir': '/shared/data/pdb',
      'out_file': 'situs_maps.pickle'}

# convert to density maps, save to situs_maps.pickle
import aitom.structure.pdb.situs_pdb2vol__batch as SPS
import pickle
re = SPS.batch_processing(op)
with open(op['out_file'], 'wb') as f:
    pickle.dump(re, f, protocol=-1)


'''
The density maps in situs_maps.pickle have different sizes for different structures,
use resize_center_batch_dict() to change them into same size
'''
import aitom.io.file as IF
ms = IF.pickle_load('situs_maps.pickle')
ms = {_: ms[_][10.0][10.0]['map'] for _ in ms}

import aitom.image.vol.util as IVU
ms = IVU.resize_center_batch_dict(vs=ms, cval=0.0)


