# coding:UTF-8
import os
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pprint
import sys
import math


def get_coord_array(path, file_name):
    """
    get atom coord as an array
    Function: get coord array of all atoms in a pdb file

    :param
        path: the path of pdb file of all proteins
        file_name: the file name of ****.pdb

    :return:
        atom coord array
        [[x0,y0,z0],
        [x1,y1,z1],
        ... ,
        [xn,yn,zn]]
    """
    parser = PDBParser(PERMISSIVE=1)
    structure_id = file_name.split('.')[0]
    path_file_name = path + file_name
    structure = parser.get_structure(structure_id, path_file_name)

    atom_coord_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coord = atom.get_coord()
                    atom_coord_list.append(atom_coord)
    atom_coord_array = np.array(atom_coord_list)
    # print file_name,'atom_coo_array\n',atom_coord_array,'\n'
    # print('get_coord_array DONE!\t', path, ": ", file_name)
    return atom_coord_array


def dist_Eur(vecA, vecB):
    return np.sqrt(sum(np.power((vecA - vecB), 2)))


def dist_Eur_array(array, origin):
    power_result = np.power((array - origin), 2)
    sum_power = power_result.sum(axis=1)  # axis = 1 row; axis = 0 column
    dist_array = np.sqrt(sum_power)
    maxdist = dist_array.max()
    dist_Eur_dic = {'dist_array': dist_array,
                    'maxdist': round(maxdist, 4)}
    return dist_Eur_dic


# PDB_ori_path = './pdbfile/'

def pdb2ball_single(PDB_ori_path='../IOfile/pdbfile/', show_log=0):
    """
    :param
        PDB_ori_path: this is the path that save all the original pdb file

    :return:
        return a dictionary 'pdb_dict', format as follows:
        {'pdb_id':
            {'pdb_id': ****, 'atom_number': ****, 'center': ****, 'radius': ****}
            'pdb_id':  {'pdb_id': ****, 'atom_number': ****, 'center': ****, 'radius': ****}
            ...}
    """
    if show_log != 0:
        print('start convert pdb file to single ball')

    pdb_dict = {}
    for file in os.listdir(PDB_ori_path):
        if file != '.DS_Store':
            # get pdb id of each protein
            # print(file)
            pdb_id = file[0:4]

            # get atom number of each protein, may be useful when calculating mass
            atom_coord_array = get_coord_array(PDB_ori_path, file)
            atom_number = len(atom_coord_array[0])

            # get center coordinate of each protein, this is the center of the boundary ball
            '''
            atom_coord_array              xyz_coord_array

            [[x0,y0,z0],               [[x0,x1,x2, ... , xn],
             [x1,y1,z1],       TO       [y0,y1,y2, ... , yn],
                 ... ,                  [z0,z1,z2, ... , zn]]
             [xn,yn,zn]]
            '''
            xyz_coord_array = atom_coord_array.T
            xyz_coord_min = np.min(xyz_coord_array, axis=1)
            xyz_coord_max = np.max(xyz_coord_array, axis=1)
            # print(xyz_coord_min)
            # print(xyz_coord_max)
            xcenter = 0.5 * (xyz_coord_min[0] + xyz_coord_max[0])
            ycenter = 0.5 * (xyz_coord_min[1] + xyz_coord_max[1])
            zcenter = 0.5 * (xyz_coord_min[2] + xyz_coord_max[2])
            xcenter = round(xcenter, 4)
            ycenter = round(ycenter, 4)
            zcenter = round(ycenter, 4)
            center_box = [xcenter, ycenter, zcenter]
            # print (center_box)
            # center = np.mean(xyz_coord_array, axis=1)  # this center is not accurate

            # get radius of the boundary ball
            radius = dist_Eur_array(atom_coord_array, center_box)['maxdist']

            tmp_dict = {'pdb_id': pdb_id,
                        'atom_number': atom_number,
                        'center': (center_box, 4),
                        'radius': (radius, 4)}

            # print('pdb 2 single ball: DONE!\t', tmp_dict)
            # print('==========================================\n\n\n')

            # save in a pdb_dict
            pdb_dict[file[0:4]] = tmp_dict
        else:
            pass
    if show_log != 0:
        # print the result
        dic_print = pprint.PrettyPrinter(indent=4)
        dic_print.pprint(pdb_dict)
        print('pdb 2 single ball: All File Done!\n\n')
    return pdb_dict


if __name__ == '__main__':
    try:
        pdb2ball_single(sys.argv[1])
    except:
        pdb2ball_single()
