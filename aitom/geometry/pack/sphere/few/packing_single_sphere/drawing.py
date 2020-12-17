from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import numpy as np
import json


def show_center_img(x, y, z):
    print("Show distribution img of protein's center")
    # create project
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r')  # draw

    ax.set_zlabel('Z')  # aix
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


def show_sum_img(sumlist, protein_num, pdb_id):
    # process sumlist
    stride = int(len(sumlist) / protein_num)
    newlist = process_sumlist(sumlist, protein_num, stride)
    print('Show sum img')
    ax = plt.subplot()
    for line in range(protein_num):
        ax.plot(newlist[stride * line:stride * (line + 1)], label=pdb_id[line].upper())
    plt.legend()
    plt.title('Loss of protein')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')


def process_sumlist(sumlist, protein_num, stride):
    newlist = [0 for _ in range(len(sumlist))]
    for ii in range(len(sumlist)):
        temp_remainder = ii % protein_num
        temp_quotient = int((ii - temp_remainder) / protein_num)
        newlist[temp_remainder * stride + temp_quotient] = sumlist[ii]
    return newlist


def get_packing_and_plot_ball(optimal_result, boundary_shpere):
    """
    optimal_result is data read from json file by instruction "json.load(file)"
    boundary_shpere is a list of the radius of each protein
    """
    pdb_id = optimal_result['pdb_id']
    x_loc = optimal_result['x']
    y_loc = optimal_result['y']
    z_loc = optimal_result['z']

    radius_list = []
    center_list = [x_loc, y_loc, z_loc]
    for ii in range(len(pdb_id)):
        radius_list.extend([boundary_shpere[pdb_id[ii]]['radius'][0]])
    print(radius_list)
    print(center_list)

    drawing_center_with_ball(radius_list, center_list)


def drawing_center_with_ball(radius_list, loc_list):
    """
    for loc_list row 0, 1, 2 of the matrix represent x,y,z,
    column 0,1,... represent location of each protein
    """
    # draw one ball
    center = [loc_list[0][0], loc_list[1][0], loc_list[2][0]]
    radius = radius_list[0]
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_list = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y_list = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z_list = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    # draw all ball
    for k in range(1, len(radius_list)):
        center = [loc_list[0][k], loc_list[1][k], loc_list[2][k]]
        radius = radius_list[k]
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        x_list = np.concatenate([x, x_list], axis=1)
        y_list = np.concatenate([y, y_list], axis=1)
        z_list = np.concatenate([z, z_list], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_list, y_list, z_list, rstride=4, cstride=4, color='b')
    plt.show()


def get_json_and_plot_ball(file):
    """
    input the path of the json file, then the function will read information in
    json file and call the drawing function
    """
    with open(file) as f:
        packing_result = json.load(f)
        pdb_id = packing_result['optimal_result']['pdb_id']  # list of the protein id
        x_loc = packing_result['optimal_result']['x']  # location of each protein
        y_loc = packing_result['optimal_result']['y']
        z_loc = packing_result['optimal_result']['z']

    radius_list = []
    center_list = [x_loc, y_loc, z_loc]  # combine x,y,z lists into one list
    for ii in range(len(pdb_id)):
        radius_list.extend(
            [packing_result['boundary_shpere'][pdb_id[ii]]['radius'][0]])  # list of radius of each protein
    print(radius_list)
    print(center_list)
    drawing_center_with_ball(radius_list, center_list)


if __name__ == "__main__":
    file = '../IOfile/packing.json'
    get_json_and_plot_ball(file)
