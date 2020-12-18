#!/usr/bin/env python
"""
after the peak detection, remove redundant peaks so that the distance between peaks are at least sigma1,
"""

import pickle
import numpy as N
from scipy.spatial.distance import cdist


def do_filter(pp, peak_dist_min, op=None):
    x = N.array([_['x'] for _ in pp])

    redundant_flag = N.array([False] * len(pp))
    for peak_i in range(len(pp)):
        if redundant_flag[peak_i]:
            continue

        if 'id' in pp[peak_i]:
            print('\r', peak_i, '    ', pp[peak_i]['id'], '           ')

        d = cdist(x, N.reshape(x[peak_i, :], (1, -1))).flatten()

        ind = N.where(d < peak_dist_min)[0]
        if ind.size > 1:
            for ind_t in ind:
                if ind_t <= peak_i:
                    continue  # assume the peak ids are ordered so that the smaller id has better value
                if redundant_flag[ind_t]:
                    continue
                redundant_flag[ind_t] = True

        if (op is not None) and ('top_num' in op) and (peak_i > (op['top_num'] * 2)):
            non_redundante_num = (redundant_flag[:peak_i] is False).sum()
            print(non_redundante_num, '        '),
            if non_redundante_num > op['top_num']:
                break

    pp_f = []
    for i, pp_t in enumerate(pp):
        if redundant_flag[i]:
            continue
        pp_f.append(pp_t)

        if (op is not None) and ('top_num' in op) and (len(pp_f) >= op['top_num']):
            break

    return pp_f


def main():
    import json
    with open('particle_picking_dog__filter__op.json') as f:
        op = json.load(f)
    with open(op['particle_picking_dog__op_file']) as f:
        pp_op = json.load(f)
    with open(pp_op['tomogram_info_file']) as f:
        tom_info = json.load(f)
    tom_info = {_['id']: _ for _ in tom_info}

    with open(op['peak_file']) as f:
        pp = pickle.load(f)

    tom_ids = set([_['tomogram_id'] for _ in pp])

    pp_t = []
    for tom_id in tom_ids:
        print('processing tomogram', tom_id)

        r = pp_op['sigma1'] / tom_info[tom_id]['voxel_spacing']

        pp_tom = [_ for _ in pp if (_['tomogram_id'] == tom_id)]
        # order peaks according to ids,  assume the peak ids are ordered so that the smaller id has better value
        pp_tom = sorted(pp_tom, key=lambda _: _['id'])

        pp_t.extend(do_filter(pp_tom, peak_dist_min=2 * r, op=op))

    print('peak number reduced from', len(pp), 'to', len(pp_t))

    with open(op['out_file'], 'w') as f:
        pickle.dump(pp_t, f, protocol=-1)


if __name__ == '__main__':
    main()
