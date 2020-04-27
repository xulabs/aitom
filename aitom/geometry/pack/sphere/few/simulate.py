import pdb2ball_single as P2B
import random_select as RS
import packing as PK
import pprint
import sys

def packing_with_target(target_protein = '1bxn', random_protein_number = 4, iteration=501, PDB_ori_path = './pdbfile/', step=1, show_img= 1, show_log = 1):
    # convert pdb file into single ball and get the center and radius of this ball.
    boundary_shpere = P2B.pdb2ball_single(PDB_ori_path = PDB_ori_path, show_log = show_log)

    # set target protein
    if show_log != 0:
        print('target protein is', target_protein,'\n\n')
    protein_name = []
    protein_name.append(target_protein)
    radii_list = []
    radii_list.append(boundary_shpere[protein_name[0]]['radius'])

    # select random proteins
    random_protein = RS.get_random_protein(boundary_shpere,protein_number = random_protein_number,show_log= show_log)

    # get important info
    info = RS.get_radius_and_id(random_protein, radii_list = radii_list, protein_name = protein_name, show_log = show_log)
    radius_list = info['radius_list']
    protein = info['protein_key']

    # set box
    box_size = PK.get_box_size(radius_list,show_log= show_log)  # obtain box size

    # random run multiple times and return the optimal result
    dict_out = {}
    sum_list = []
    for i in range(2):  # try n times and get the optimal solution
        print('Round', i+1)
        # initialization
        location = PK.initialization(radius_list, box_size, show_img = 0, show_log = show_log)
        # packing
        dict = PK.do_packing(radius_list, location, iteration=iteration, step=step, show_img= show_img, show_log= show_log)
        dict_out[i] = dict

        # save result
        sum_list.append(dict['sum'])
        print('sum:\t',dict['sum'], '\tgrad:\t',dict['grad'],'\n\n')

    # choose a best result
    index = sum_list.index(min(sum_list))
    min_dict = dict_out[index]
    min_dict['pdb_id'] = protein
    min_dict['box_size'] = box_size
    print('The following is the optimal solution:')
    dic_print = pprint.PrettyPrinter(indent=4)
    dic_print.pprint(min_dict)

    # save general information
    general_info = {}
    general_info['target'] = target_protein
    general_info['random_protein_number'] = random_protein_number
    general_info['iteration'] = iteration
    general_info['step'] = step
    general_info['box_size'] = box_size

    # save return information
    packing_result = {}
    packing_result['general_info'] = general_info
    packing_result['boundary_shpere'] = boundary_shpere
    packing_result['optimal_result'] = min_dict

    return packing_result


if __name__ == '__main__':
    try:
        packing_with_target(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[2])
    except:
        packing_with_target()