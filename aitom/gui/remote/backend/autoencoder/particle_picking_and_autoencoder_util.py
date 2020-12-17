import os
import shutil
import uuid

import matplotlib.pyplot as plt
import numpy as np

import aitom.image.vol.util as im_vol_util
import aitom.io.file as AIF
import aitom.io.file as io_file
from aitom.filter.gaussian import smooth


class ParticlePicking():
    def __init__(self, file_path, output_dir):
        self.path = file_path
        self.dump_path = output_dir

    def select_sigma(self):
        mrc_header = io_file.read_mrc_header(self.path)
        voxel_spacing_in_nm = mrc_header['MRC']['xlen'] / mrc_header['MRC']['nx'] / 10
        # In general, 7 is optimal sigma1 val in nm according to the paper and sigma1 should at least be 2
        return max(7 / voxel_spacing_in_nm, 2)

    def dump_subvol(self, picking_result):
        from aitom.classify.deep.unsupervised.autoencoder.autoencoder_util import peaks_to_subvolumes
        subvols_loc = os.path.join(self.dump_path, "demo_single_particle_subvolumes.pickle")
        a = io_file.read_mrc_data(self.path)
        d = peaks_to_subvolumes(im_vol_util.cub_img(a)['vt'], picking_result, 32)
        io_file.pickle_dump(d, subvols_loc)
        print("Save subvolumes .pickle file to:", subvols_loc)

    def view_tomo(self, sigma=2, R=10):
        # d = {v_siz:(32,32,32), vs:{uuid0:{center, v, id}, uuid1:{center, v, id} ... }}
        subvols_loc = os.path.join(self.dump_path, "demo_single_particle_subvolumes.pickle")
        d = io_file.pickle_load(subvols_loc)
        a = io_file.read_mrc_data(self.path)

        if 'self.centers' not in dir():
            centers = []
            uuids = []
            for k, v in d['vs'].items():
                if v['v'] is not None:
                    centers.append(v['center'])
                    uuids.append(k)
            self.centers = centers
            self.uuids = uuids

        # denoise
        a_smooth = smooth(a, sigma)

        for slice_num in range(a_smooth.shape[2]):
            centers = np.array(centers)

            slice_centers = centers[(centers[:, 2] - slice_num) ** 2 < R ** 2]
            img = a_smooth[:, :, slice_num]
            plt.rcParams['figure.figsize'] = (15.0, 12.0)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.axis('off')
            for center_num in range(len(slice_centers)):
                y, x = slice_centers[center_num][0:2]
                r = np.sqrt(R ** 2 - (slice_centers[center_num][2] - slice_num) ** 2)
                circle = plt.Circle((x, y), r, color='b', fill=False)
                plt.gcf().gca().add_artist(circle)
            ax_u = ax.imshow(img, cmap='gray')

    def view_subtom(self, subvol_num, sigma=2, R=10):
        subvols_loc = os.path.join(self.dump_path, "demo_single_particle_subvolumes.pickle")
        d = io_file.pickle_load(subvols_loc)
        a = io_file.read_mrc_data(self.path)

        # denoise
        a_smooth = smooth(a, sigma)

        if 'self.centers' not in dir():
            centers = []
            uuids = []
            for k, v in d['vs'].items():
                if v['v'] is not None:
                    centers.append(v['center'])
                    uuids.append(k)
            self.centers = centers
            self.uuids = uuids

        y, x, z = self.centers[subvol_num]
        img = a_smooth[:, :, z]
        plt.rcParams['figure.figsize'] = (10.0, 8.0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        circle = plt.Circle((x, y), R, color='b', fill=False)
        plt.gcf().gca().add_artist(circle)
        plt.axis('off')
        output_str = '%d of %d, uuid = %s' % (
            subvol_num, len(centers), self.uuids[subvol_num])
        plt.title(output_str)
        ax_u = ax.imshow(img, cmap='gray')
        save_path = 'tmp_sub.png'
        # plt.imsave(save_path, img, cmap='gray')
        plt.savefig(save_path)
        return save_path

    def select(self, remove_particles, pick_num):
        d = io_file.pickle_load(os.path.join(self.dump_path, "demo_single_particle_subvolumes.pickle"))
        subvols_loc = os.path.join(self.dump_path, "selected_demo_single_particle_subvolumes.pickle")
        particles_num = pick_num
        result = {'v_siz': d['v_siz'], 'vs': {}}
        remove_particles = np.array(remove_particles)
        # d = {v_siz:(32,32,32), vs:{uuid0:{center, v, id}, uuid1:{center, v, id} ... }}

        for i in range(len(self.centers)):
            if i in remove_particles:
                continue
            uuid_i = self.uuids[i]
            result['vs'][uuid_i] = d['vs'][uuid_i]
            if len(result['vs']) >= particles_num:
                break
        assert len(result['vs']) == particles_num
        # subvols_loc = './tmp/picking/selected_demo_single_particle_subvolumes.pickle'
        AIF.pickle_dump(result, subvols_loc)
        print("Save subvolumes .pickle file to:", subvols_loc)


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def view_clusters(out_dir):
    from matplotlib.image import imread
    plt.clf()
    fig_dir = os.path.join(out_dir, 'clus-center/fig')
    for filename in os.listdir(fig_dir):
        img = imread(os.path.join(fig_dir, filename))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axis('off')
        ax_u = ax.imshow(img, cmap='gray')
    plt.savefig('tmp_ae.png')


def combine_subtom(out_dir, pickle_path):
    subvols_loc = os.path.join(out_dir, 'selected_demo_single_particle_subvolumes.pickle')
    pickle_data = AIF.pickle_load(pickle_path)
    d = AIF.pickle_load(subvols_loc)
    subvols = []
    for v in d['vs'].values():
        if v['v'] is not None:
            subvols.append(v['v'])

    subtom = pickle_data['1KP8_data'] + pickle_data['1KP8_data'] + subvols[:100]
    print('Total subtomograms: ', len(subtom))
    subvols_loc = os.path.join(out_dir, 'subvolumes.pickle')
    d = {}
    d['v_siz'] = np.array([32, 32, 32])
    d['vs'] = {}
    labels = {}
    for i in range(len(subtom)):
        uuid_i = str(uuid.uuid4())
        d['vs'][uuid_i] = {}
        d['vs'][uuid_i]['center'] = None
        d['vs'][uuid_i]['id'] = uuid_i
        d['vs'][uuid_i]['v'] = subtom[i]
        d['vs'][uuid_i]['label'] = int(i / 100)

    AIF.pickle_dump(d, subvols_loc)
    print("Save subvolumes .pickle file to:", subvols_loc)
