import random
import pprint


def get_random_protein(boundary_shpere, protein_number = 5, show_log = 0):
    '''

    :param boundary_shpere: the dictionary of the macromolucules' boundary sphere
    :param protein_number: The number of the random neighbors
    :param show_log: print the log or not
    :return: dictinary of all select macromolecules,

    random_protein =

            {
                'pdb_id':  {'pdb_id': ****, 'atom_number': ****, 'center': ****, 'radius': ****}
                'pdb_id':  {'pdb_id': ****, 'atom_number': ****, 'center': ****, 'radius': ****}
                ...
            }

    '''
    print('Start to select', protein_number, 'random proteins')
    random_protein = {}
    for keys in boundary_shpere.keys():
        random_key = random.sample(boundary_shpere.keys(), protein_number)  # random dic, the second parameter is the number
    for i in range(len(random_key)):
        random_protein[random_key[i]] = boundary_shpere[random_key[i]]
    if show_log != 0:
        dic_print = pprint.PrettyPrinter(indent=4)
        dic_print.pprint(random_protein)
    print('Get random protein: Done!\n\n')
    return random_protein

def get_radius_and_id(random_protein, radii_list = [], protein_name = [], show_log = 0):
    '''

    :param random_protein: the dict of the random proteins boundary sphere
    :param radii_list: the initial radii list
    :param protein_name: the initial protein_name list
    :param show_log: print the log or not
    :return: the dictionary of the radii and pdb_id of the selected macromolecules

    random_dic = {
                    'protein_key' : [pdb_id 1, pdb_id 2, ..., pdb_id n]
                    'radius_list' : [R1, R2, ... , Rn]
                }

    '''
    # get all radii
    print('Start to get the radius of each protein')
    for protein in random_protein:
        radii_list.append(random_protein[protein]['radius'])
        protein_name.append(random_protein[protein]['pdb_id'])

    random_dic = {}
    random_dic['protein_key'] = protein_name
    random_dic['radius_list'] = radii_list

    if show_log != 0:
        dic_print = pprint.PrettyPrinter(indent=4)
        dic_print.pprint(random_dic)

    print('Get all radius value: Done!\n\n')
    return random_dic
