import sys

import aitom.geometry.ang_loc as GAL
import aitom.geometry.rotate as GR
import json
import numpy as np

sys.path.append("..")

# set parameters for the simulation
op = {'model': {'missing_wedge_angle': 30,
                'SNR': 500000000},
      'ctf': {'pix_size': 1.0,
              'Dz': -5.0,
              'voltage': 300,
              'Cs': 2.0,
              'sigma': 0.4}}
num = 1
output = {
    'initmap': {'mrc': '../IOfile/initmap/mrc/initmap{}.mrc'.format(num),
                'png': '../IOfile/initmap/png/initmap{}.png'.format(num),
                'trim': '../IOfile/initmap/trim/initmap{}T.mrc'.format(num)},
    'packmap': {'mrc': '../IOfile/packmap/mrc/packmap{}.mrc'.format(num),
                'png': '../IOfile/packmap/png/packmap{}.png'.format(num),
                'trim': '../IOfile/packmap/trim/packmap{}T.mrc'.format(num)},
    'tomo': {'mrc': '../IOfile/tomo/mrc/tomo{}.mrc'.format(num),
             'png': '../IOfile/tomo/png/tomo{}.png'.format(num),
             'trim': '../IOfile/tomo/trim/tomo{}T.mrc'.format(num)},
    'json': {'pack': '../IOfile/json/packing{}.json'.format(num),
             'target': '../IOfile/json/target{}.json'.format(num)}}


def random_rotate(v):
    """randomly rotate and translate v"""
    angle = GAL.random_rotation_angle_zyz()
    vr = GR.rotate(v, angle=angle, default_val=0.0)  # loc_r is none
    # print('angle:', angle)
    # print('loc_max:', loc_max)
    # print('loc_r:',type(loc_r),loc_r)
    return vr, angle


def angle_rotate(v, angle):
    vr = GR.rotate(v, angle=angle, default_val=0.0)
    return vr


def merge_map(v, protein_name, x, y, z, box_size):
    def add_map(hugemap, map, center):
        assert (center.shape == (3,))
        x0 = int(center[0]) - map.shape[0] / 2
        y0 = int(center[1]) - map.shape[1] / 2
        z0 = int(center[2]) - map.shape[2] / 2
        for i in range(map.shape[0]):
            x = x0 + i
            if x < 0 or x >= hugemap.shape[0]:
                continue
            for j in range(map.shape[1]):
                y = y0 + j
                if y < 0 or y >= hugemap.shape[1]:
                    continue
                for k in range(map.shape[2]):
                    z = z0 + k
                    if z < 0 or z >= hugemap.shape[2]:
                        continue
                    hugemap[x][y][z] += map[i][j][k]
        return hugemap

    hugemap = np.zeros((box_size, box_size, box_size), dtype='float32')
    angle_list = []
    for i in range(len(protein_name)):
        n = protein_name[i]
        vr, angle = random_rotate(v[n])
        angle_list.append(angle)
        hugemap = add_map(hugemap, vr, np.array([x[i], y[i], z[i]]))
    return hugemap, angle_list


def trim_margin(hugemap):
    def pad_to_cub(a, pad_value=0):
        padded = pad_value * np.ones(3 * [max(a.shape)], dtype=a.dtype)
        x_begin = (padded.shape[0] - a.shape[0]) / 2
        y_begin = (padded.shape[1] - a.shape[1]) / 2
        z_begin = (padded.shape[2] - a.shape[2]) / 2
        padded[x_begin:x_begin + a.shape[0], y_begin:y_begin + a.shape[1], z_begin:z_begin + a.shape[2]] = a
        return padded

    x, y, z = np.nonzero(hugemap)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    z_min = np.min(z)
    z_max = np.max(z)
    trimmed = hugemap[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    # resize to cub
    trimmed_cub = pad_to_cub(trimmed)
    return trimmed_cub


def trim_target(hugemap, target_center, target_size=32, loc_r=None):
    # volume size = 30*30*30
    side = target_size
    if loc_r is None:
        loc_proportion = 0.5
        loc_max = np.array([side, side, side], dtype=float) * loc_proportion
        loc_r = (np.random.random(3) - 0.5) * loc_max  # -0.5 ~ 0.5

    loc_r = loc_r.astype(np.int16)
    target_center = target_center.astype(np.int16)
    trim_center = target_center + (-loc_r)
    x_begin = max(trim_center[0] - side / 2, 0)
    y_begin = max(trim_center[1] - side / 2, 0)
    z_begin = max(trim_center[2] - side / 2, 0)
    targetmap = hugemap[x_begin: x_begin + side, y_begin: y_begin + side, z_begin: z_begin + side]
    return targetmap, loc_r


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# if __name__ == '__main__':
#     # read density map from mrc
#     import mrcfile
#     import iomap as IM
#     rootdir = '../IOfile/map_single'
#     v = IM.readMrcMapDir(rootdir)
#
#     # get packing info
#     import sys
#     sys.path.append("..")
#     import packing_single_sphere.simulate as SI
#     import numpy as np
#     import packing_single_sphere.simulate as SI
#     target_name = '1bxn'
#     packing_result = SI.packing_with_target(target_protein=target_name, random_protein_number=4)
#     protein_name = packing_result['optimal_result']['pdb_id']
#     x = packing_result['optimal_result']['x']/10
#     y = packing_result['optimal_result']['y']/10
#     z = packing_result['optimal_result']['z']/10
#     print('initialization',packing_result['optimal_result']['initialization'])
#     x0 = np.array(packing_result['optimal_result']['initialization'][0])/10
#     y0 = np.array(packing_result['optimal_result']['initialization'][1])/10
#     z0 = np.array(packing_result['optimal_result']['initialization'][2])/10
#     box_size = packing_result['general_info']['box_size']/10
#
#     # merge map to hugemap, save random angle in packing_result
#     import map_tomo.merge_map as MM
#     initmap,init_angle_list = MM.merge_map(v, protein_name, x0, y0, z0, box_size)
#     packmap,pack_angle_list = MM.merge_map(v, protein_name, x, y, z, box_size)
#     packing_result['optimal_result']['initmap_rotate_angle'] = init_angle_list
#     packing_result['optimal_result']['packmap_rotate_angle'] = pack_angle_list
#     IM.map2mrc(initmap, output['initmap']['mrc'])
#     IM.map2mrc(packmap, output['packmap']['mrc'])
#     # save packing info
#     with open(output['json']['pack'],'w') as f:
#         json.dump(packing_result, f, cls=MM.NumpyEncoder)
#
#     # trim hugemap
#     trim_initmap = trim_margin(initmap)
#     trim_packmap = trim_margin(packmap)
#     print('initmap shape',initmap.shape)
#     print('trimmed shape',trim_initmap.shape)
#     print('packmap shape',packmap.shape)
#     print('trimmed shape',trim_packmap.shape)
#     IM.map2mrc(trim_initmap, output['initmap']['trim'])
#     IM.map2mrc(trim_packmap, output['packmap']['trim'])
#
#     # save huge/trim map to png
#     IM.map2png(initmap, output['initmap']['png'])
#     IM.map2png(packmap, output['packmap']['png'])
