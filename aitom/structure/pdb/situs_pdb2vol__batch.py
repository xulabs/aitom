import uuid
from . import situs_pdb2vol as SP
from six.moves import input as raw_input


def batch_processing(op):
    """
    automatically scan pdb files and convert them to density maps, and save in to a pickle file.
    This is done in parallel Alternatively, save in matlab format that is same as Bsoft.pdb2em_batch_convert_test().
    """
    import os
    # walk through every subdir, find all pdb files
    extension = '.pdb'
    pdb_path = {}
    for root, sub_folders, files in os.walk(op['pdb_dir']):
        for file_t in files:
            if not file_t.endswith(extension):
                continue

            pdb_id = file_t[:-4]

            assert (pdb_id + extension) == file_t
            assert pdb_id not in pdb_path  # the pdb_id must be unique

            pdb_path[pdb_id] = os.path.join(root, file_t)

    if 'pdb_id_selected' in op:
        pdb_path = {_: pdb_path[_] for _ in (set(pdb_path.keys()) & set(op['pdb_id_selected']))}

    print('generating maps for ', len(pdb_path), 'structures')

    import copy
    ts = {}
    for pdb_id in pdb_path:
        for spacing in op['spacing_s']:
            for resolution in op['resolution_s']:

                op_t = copy.deepcopy(op)
                op_t['pdb_id'] = pdb_id
                op_t['pdb_file'] = pdb_path[pdb_id]

                assert 'resolution' not in op_t
                op_t['resolution'] = resolution
                assert 'spacing' not in op_t
                op_t['spacing'] = spacing

                ts[uuid.uuid4()] = {'func': SP.convert, 'kwargs': {'op': op_t}}

    import aitom.parallel.multiprocessing.util as TPMU
    cre_s = TPMU.run_batch(ts)

    re = {}
    for cre in cre_s:
        pdb_id = cre['result']['pdb_id']
        resolution = cre['result']['resolution']
        spacing = cre['result']['spacing']

        if pdb_id not in re:
            re[pdb_id] = {}

        if spacing not in re[pdb_id]:
            re[pdb_id][spacing] = {}

        assert resolution not in re[pdb_id][spacing]

        re[pdb_id][spacing][resolution] = cre['result']

    return re


def display(path):
    import pickle
    with open(path) as f:
        m = pickle.load(f)

    import matplotlib.pyplot
    import aitom.image.vol.util as VU
    for pdb_id in m:
        for voxel_size in m[pdb_id]:
            for reolution in m[pdb_id][voxel_size]:
                print('pdb_id', pdb_id, 'voxel_size', voxel_size, 'reolution', reolution)
                VU.dsp_cub(m[pdb_id][voxel_size][reolution]['map'])

            raw_input('press enter')
            matplotlib.pyplot.close('all')


def test0():
    op = {'situs_pdb2vol_program': '/opt/local/img/em/et/util/situs/Situs_2.7.2/bin/pdb2vol',
          'spacing_s': [10.0],
          'resolution_s': [10.0],
          'pdb_dir': './pdb',
          'out_file': 'situs_maps.pickle'}

    re = batch_processing(op)

    import pickle
    with open(op['out_file'], 'wb') as f:   pickle.dump(re, f, protocol=-1)


def main():
    import sys
    import json

    op_file = 'situs_pdb2vol__batch__op.json'

    with open(op_file) as f:
        op = json.load(f)

    re = batch_processing(op)

    # display(op['out_file'])
    import pickle
    with open(op['out_file'], 'wb') as f:
        pickle.dump(re, f, protocol=-1)


if __name__ == '__main__':
    main()
