"""
If you want to see the output image(location, loss, etc.) of the simulating process
please uncomment the line 4, line 134-139.
"""
from . import pdb2ball_single as P2B
from . import random_select as RS
from . import packing as PK
# from . import drawing as DR
import pprint
import sys

packing_op = {'target': '1bxn',
              'random_protein_number': 4,
              'PDB_ori_path': '../IOfile/pdbfile/',
              'iteration': 5001,
              'step': 1,
              'show_img': 1,
              'show_log': 1,
              'boundary_shpere': None}


def packing_with_target(packing_op):
    """
    :param
        target_protein: the name of the target macromolecule
        random_protein_number: the number of random neighbors
        iteration: the number of iteration time
        PDB_ori_path: the path to all pdb files
        step: the moving step(parameter to chotrol how long a macromolecules will move) in each iteration
        show_img: draw image or not
        show_log: print log or not
    :return:
        the optimal packing result and other info
        packing_result = {
            'general_info': {
                'target': name of the target macromolecules
                'random_protein_number'
                'iteration'
                'step'
                'box_size'}
            'boundary_shpere': a dictionary format the same as the output in pdb2ball_single {
                'pdb_id': {'pdb_id': ****, 'atom_number': ****, 'center': ****, 'radius': ****}
                'pdb_id': {'pdb_id': ****, 'atom_number': ****, 'center': ****, 'radius': ****}
                ...}
            'optimal_result': {
                'pdb_id': a list of all macromolecules' pdb_id
                'box_size': a number, the size of the simulation field
                # the following information is the same as the output of do_packing()
                'sum' : a number, final value of the loss
                'grad' : a number, final valus of the grad of the loss
                'x' : list, final location of all macromolecules in the simulation field
                'y' : list, final location of all macromolecules in the simulation field
                'z' : list, final location of all macromolecules in the simulation field
                'sum_list' : list, the sumlist of all the macromolecules, is used to draw the img of loss
                    in each iteration}}
    """
    # convert pdb file into single ball and get the center and radius of this ball.
    # boundary_shpere = P2B.pdb2ball_single(PDB_ori_path=packing_op['PDB_ori_path'], show_log=packing_op['show_log'])

    # set target protein
    if packing_op['show_log'] != 0:
        print('target protein is', packing_op['target'], '\n\n')
    protein_name = [packing_op['target']]
    radii_list = [packing_op['boundary_shpere'][protein_name[0]]['radius']]

    # select random proteins
    random_protein = RS.get_random_protein(packing_op['boundary_shpere'],
                                           protein_number=packing_op['random_protein_number'],
                                           show_log=packing_op['show_log'])

    # get important info
    info = RS.get_radius_and_id(random_protein, radii_list=radii_list, protein_name=protein_name,
                                show_log=packing_op['show_log'])
    radius_list = info['radius_list']
    protein = info['protein_key']

    # set box
    box_size = PK.get_box_size(radius_list, show_log=packing_op['show_log'])  # obtain box size

    # random run multiple times and return the optimal result
    dict_out = {}
    sum_list = []

    # try n times and get the optimal solution
    for i in range(5):
        if packing_op['show_log'] != 0:
            print('Round', i + 1)

        # initialization
        location = PK.initialization(radius_list, box_size, show_log=packing_op['show_log'])
        save_location = [tuple(location[0]), tuple(location[1]), tuple(location[2])]
        # print('init 1',location)

        # packing
        dict = PK.do_packing(radius_list, location, iteration=packing_op['iteration'], step=packing_op['step'],
                             show_log=packing_op['show_log'])
        save_location = [list(save_location[0]), list(save_location[1]), list(save_location[2])]
        dict['initialization'] = save_location
        dict_out[i] = dict
        # print('init x', dict['initialization'][0])
        # print('init y', dict['initialization'][1])
        # print('init z', dict['initialization'][2])
        # print('final x',dict['x'])
        # print('final y',dict['y'])
        # print('final z',dict['z'])

        # save result
        sum_list.append(dict['sum'])
        if packing_op['show_log'] != 0:
            print('sum:\t', dict['sum'], '\tgrad:\t', dict['grad'], '\n\n')

    # choose a best result
    index = sum_list.index(min(sum_list))
    min_dict = dict_out[index]
    min_dict['pdb_id'] = protein
    min_dict['box_size'] = box_size

    # delete sum_list and print the optimal result
    if packing_op['show_log'] != 0:
        print('The following is the optimal solution:')
        sum_list = min_dict.pop('sum_list')
        dic_print = pprint.PrettyPrinter(indent=4)
        dic_print.pprint(min_dict)
        # add back sum_list
        min_dict['sum_list'] = sum_list

    # show image
    # if packing_op['show_img'] != 0:
    #    DR.show_center_img(min_dict['initialization'][0], min_dict['initialization'][1], min_dict['initialization'][2])
    #    DR.show_center_img(min_dict['x'], min_dict['y'], min_dict['z'])
    #    DR.show_sum_img(min_dict['sum_list'], len(radius_list), protein_name)
    #    DR.get_packing_and_plot_ball(min_dict, boundary_shpere)

    # save general information
    general_info = {'target': packing_op['target'],
                    'random_protein_number': packing_op['random_protein_number'],
                    'iteration': packing_op['iteration'],
                    'step': packing_op['step'],
                    'box_size': box_size}

    # save return information
    packing_result = {'general_info': general_info,
                      'boundary_shpere': packing_op['boundary_shpere'],
                      'optimal_result': min_dict}

    return packing_result


if __name__ == '__main__':
    # convert pdb file into single ball and get the center and radius of this ball.
    packing_op['boundary_shpere'] = P2B.pdb2ball_single(PDB_ori_path='../IOfile/pdbfile/', show_log=0)
    try:
        packing_with_target(sys.argv[1])
    except:
        packing_with_target(packing_op)
