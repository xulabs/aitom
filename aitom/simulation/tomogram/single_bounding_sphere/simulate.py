import pack
import math
import mrcfile
import copy
from random import shuffle
import numpy as np
# import matplotlib.pyplot as plt
import miniball
from Bio.PDB import *
# from mpl_toolkits.mplot3d import Axes3D
# from itertools import product, combinations
from mayavi.mlab import *

from . import boundingSphere
import aitom.io.file as AIF


def get_mass(path):
    atom_count = 0
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure("X", path)
    for atom in structure.get_atoms():
        atom_count += 1

    return atom_count
    # visited = {}
    # for line in open(path):
    #     l = line.split()
    #     id = l[0]
    #     if id == 'ATOM':
    #         type = l[2]
    #         if type == 'CA':
    #             residue = l[3]
    #             type_of_chain = l[4]
    #             atom_count = int(l[5])
    #             position = l[6:8]
    #             if atom_count >= 0:
    #                 if type_of_chain not in visited:
    #                     visited[type_of_chain] = 1
    #                     return atom_count
    #                     # print (residue,type_of_chain,atom_count,' '.join(position))
    # return atom_count


def find_bounding_sphere(mrc, L):
    v = AIF.read_mrc(mrc)

    points = []
    density_max = v.max()
    contour_level = L * density_max
    for ijk in np.ndindex(v.shape):
        if v[ijk] >= contour_level:
            points.append([float(ijk[0]), float(ijk[1]), float(ijk[2])])

    points = np.asarray(points)

    # return boundingSphere.find_min_bounding_sphere(points)
    return miniball.Miniball(points)


def get_distribution(proteins, Ntot):
    num = len(proteins)
    f = np.random.random(num)
    f /= f.sum()
    dist = np.random.multinomial(Ntot, f)
    conf = []

    for i in range(len(dist)):
        protein = proteins[i]
        mrc = 'mrc1/' + protein + '.mrc'
        pdb = 'pdb/' + protein + '.pdb'
        R, C = find_bounding_sphere(mrc, 0.2)
        # mass = np.random.randint(0,100)
        mass = get_mass(pdb)

        num_occurrence = dist[i]

        for j in range(num_occurrence):
            conf.append({'r': R, 'mass': mass, 'id': protein})

    return conf


def dist(proteins, data):
    conf = []
    for protein in proteins:
        for j in range(20):
            pro = copy.deepcopy(data[protein])
            pro['mass'] += j
            conf.append(pro)

    return conf


leng = 1200

p1a1s = {'r': 3.9597, 'mass': 35098.5, 'id': '1A1S', 'c': 'b'}
p1eqr = {'r': 8.2916, 'mass': 198063.6, 'id': '1EQR', 'c': 'g'}
p1gyt = {'r': 12.2575, 'mass': 661785.41, 'id': '1GYT', 'c': 'r'}
p1kyi = {'r': 10.2705, 'mass': 834906.19, 'id': '1KYI', 'c': 'c'}
p1vpx = {'r': 9.0134, 'mass': 516090.03, 'id': '1VPX', 'c': 'm'}
p1w6t = {'r': 4.767, 'mass': 97702.8, 'id': '1W6T', 'c': 'y'}
# p2awb = {'r': 11.4888, 'mass': 4293422.15, 'id': '2AWB'}
p2byu = {'r': 6.6332, 'mass': 148515.59, 'id': '2BYU', 'c': 'k'}
p2gls = {'r': 8.3555, 'mass': 623832.89, 'id': '2GLS', 'c': 'w'}
p2ldb = {'r': 6.6077, 'mass': 172942.04, 'id': '2LDB', 'c': 'b'}
p3dy4 = {'r': 9.7921, 'mass': 705072, 'id': '3DY4', 'c': 'g'}
p1bxr = {'r': 12.0576, 'mass': 644905.85, 'id': '1BXR', 'c': 'r'}
p1f1b = {'r': 6.7334, 'mass': 103551.4, 'id': '1F1B', 'c': 'c'}
p1kp8 = {'r': 9.8714, 'mass': 810237.09, 'id': '1KP8', 'c': 'm'}
p1qo1 = {'r': 10.6771, 'mass': 448855.91, 'id': '1QO1', 'c': 'y'}
p1vrg = {'r': 6.8439, 'mass': 352304.08, 'id': '1VRG', 'c': 'k'}
p1yg6 = {'r': 6.8267, 'mass': 303872.45, 'id': '1YG6', 'c': 'w'}
p2bo9 = {'r': 6.6997, 'mass': 123008.66, 'id': '2BO9', 'c': 'b'}
p2gho = {'r': 7.6688, 'mass': 333287.4, 'id': '2GHO', 'c': 'g'}
p2h12 = {'r': 7.0397, 'mass': 298564.65, 'id': '2H12', 'c': 'r'}
p2rec = {'r': 7.0887, 'mass': 228099.61, 'id': '2REC', 'c': 'c'}

proteins = ['1a1s', '1bxr', '1eqr', '1f1b', '1gyt', '1kp8', '1kyi', '1qo1', '1vpx', '1vrg', '1w6t', '1yg6', '2bo9',
            '2byu', '2gho', '2gls', '2h12', '2ldb', '2rec', '3dy4']
data = {'1a1s': p1a1s, '1eqr': p1eqr, '1gyt': p1gyt, '1kyi': p1kyi, '1vpx': p1vpx, '1w6t': p1w6t, '2byu': p2byu,
        '2gls': p2gls, '2ldb': p2ldb, '3dy4': p3dy4, '1bxr': p1bxr, '1f1b': p1f1b, '1kp8': p1kp8, '1qo1': p1qo1,
        '1vrg': p1vrg, '1yg6': p1yg6, '2bo9': p2bo9, '2gho': p2gho, '2h12': p2h12, '2rec': p2rec}
# proteins = ['1A1S', '1BXR', '1EQR', '1FNT', '1KP8', '1LB3', '1VPX', '1W6T', '2AWB', '2BYU', '2H12', '2REC', '3GPT',
# '4V4A', '5L5A', '1AON', '1DPB', '1F1B', '1GYT', '1KYI', '1QO1', '1VRG', '1YG6', '2BO9', '2GHO', '2IDB', '3DY4',
# '3K7A', '4V4Q']
# conf = get_distribution(proteins, 400)

conf = dist(proteins, data)

# shuffle(conf)
print(conf)

cx = pack.do_packing(conf, op={'box': {'x': leng, 'y': leng, 'z': leng}, 'temprature': 3000, 'temprature decrease': 100,
                               'step': 5, 'recent_scores number': 50, 'recent_scores slope min': 0.001,
                               'min score': 0.1}, pymol_file_name='./t.pym')

l = cx['conf']
print(l)
print(cx['inside_box_num'])
print(cx['score'])
print(cx['temprature'])

arr = np.asarray(l)
np.save("particles.npy", arr)

xs = []
ys = []
zs = []
rs = []

for protein in l:
    xs.append(protein['x'][0])
    ys.append(protein['x'][1])
    zs.append(protein['x'][2])
    rs.append(protein['r'])

points3d(xs, ys, zs, rs)
show()

# def WireframeSphere(centre, radius, n_meridians=20, n_circles_latitude=None):
#     """
#     Create the arrays of values to plot the wireframe of a sphere.
#
#     Parameters
#     ----------
#     centre: array like
#         A point, defined as an iterable of three numerical values.
#     radius: number
#         The radius of the sphere.
#     n_meridians: int
#         The number of meridians to display (circles that pass on both poles).
#     n_circles_latitude: int
#         The number of horizontal circles (akin to the Equator) to display.
#         Notice this includes one for each pole, and defaults to 4 or half
#         of the *n_meridians* if the latter is larger.
#
#     Returns
#     -------
#     sphere_x, sphere_y, sphere_z: arrays
#         The arrays with the coordinates of the points to make the wireframe.
#         Their shape is (n_meridians, n_circles_latitude).
#
#     Examples
#     --------
#     >>> fig = plt.figure()
#     >>> ax = fig.gca(projection='3d')
#     >>> ax.set_aspect("equal")
#     >>> sphere = ax.plot_wireframe(*WireframeSphere(), color="r", alpha=0.5)
#     >>> fig.show()
#
#     >>> fig = plt.figure()
#     >>> ax = fig.gca(projection='3d')
#     >>> ax.set_aspect("equal")
#     >>> frame_xs, frame_ys, frame_zs = WireframeSphere()
#     >>> sphere = ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.5)
#     >>> fig.show()
#     """
#     if n_circles_latitude is None:
#         n_circles_latitude = max(n_meridians/2, 4)
#     u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
#     sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
#     sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
#     sphere_z = centre[2] + radius * np.cos(v)
#     return sphere_x, sphere_y, sphere_z


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # ax.set_aspect('equal')
# for sphere in l:
#     ax.plot_wireframe(*WireframeSphere(sphere['x'], sphere['r']), color=sphere['c'], alpha=0.5)
# fig.show()
# plt.show()
