#!/usr/bin/env python
"""
use imod to display location of a set of particles
"""

import json
import os
import subprocess
import uuid
import numpy as N


def generate_lines(x_full, rad, view_direction=0):
    l = {}
    for i in range(x_full.shape[0]):
        x = x_full[i, :].flatten().tolist()

        l[i] = {}
        for r in range(int(N.floor(rad))):
            l[i][r] = []
            d = int(N.floor(N.sqrt(rad ** 2 - r ** 2)))

            if view_direction == 0:
                # x-y plane
                l[i][r].append([[x[0] - d, x[1], x[2] + r], [x[0] + d, x[1], x[2] + r]])
                l[i][r].append([[x[0], x[1] - d, x[2] - r], [x[0], x[1] + d, x[2] - r]])
            elif view_direction == 1:
                # x-z plane
                l[i][r].append([[x[0] - d, x[1] + r, x[2]], [x[0] + d, x[1] + r, x[2]]])
                l[i][r].append([[x[0], x[1] - r, x[2] - d], [x[0], x[1] - r, x[2] + d]])
            elif view_direction == 2:
                # y-z plane
                l[i][r].append([[x[0] + r, x[1] - d, x[2]], [x[0] + r, x[1] + d, x[2]]])
                l[i][r].append([[x[0] - r, x[1], x[2] - d], [x[0] - r, x[1], x[2] + d]])
            else:
                raise Exception('view_direction')

    return l


def write_point_file(l, point_file):
    with open(point_file, 'w') as f:
        for i in l:
            contour_c = 1
            for r in l[i]:
                for c in range(len(l[i][r])):
                    print('\t', i + 1, '\t', contour_c, '\t', l[i][r][c][0][0], '\t', l[i][r][c][0][1], '\t',
                          l[i][r][c][0][2], file=f)
                    print('\t', i + 1, '\t', contour_c, '\t', l[i][r][c][1][0], '\t', l[i][r][c][1][1], '\t',
                          l[i][r][c][1][2], file=f)
                    contour_c += 1


def display_map_with_lines(l, map_file, clip_file=None, remove_intermediate_file=True):
    # save lines as text file
    point_file = os.path.join('/tmp', '3dmod-points--' + str(uuid.uuid1()))
    print('generating', point_file)
    write_point_file(l=l, point_file=point_file)

    # call point2model to convert to model file
    model_file = point_file + '.mod'
    subprocess.call(['point2model', point_file, model_file])

    assert os.path.isfile(model_file)

    # if the model file is generated, call imod to display both the density map and model file
    print('3dmod', map_file, model_file)
    subprocess.call(['3dmod', map_file, model_file])

    if remove_intermediate_file:
        import time
        time.sleep(120)

        print('deleting intermediate files')
        os.remove(point_file)
        os.remove(model_file)


def main():
    with open('particle_location_display_imod__op.json') as f:
        op = json.load(f)

    with open(op['data_json_file']) as f:
        dj = json.load(f)

    x = N.zeros((len(dj), 3))
    for i, d in enumerate(dj):
        x[i, :] = N.array(d['peak']['loc'])

    l = generate_lines(x_full=x, rad=op['rad'], view_direction=op['view_direction'])

    display_map_with_lines(l=l, map_file=op['map_file'], remove_intermediate_file=(
        op['remove_intermediate_file'] if 'remove_intermediate_file' in op else True))


if __name__ == '__main__':
    main()
