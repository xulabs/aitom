"""
a tutorial on using morphsnake and RANSC for membrane segmentation
"""

import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq

import aitom.io.file as AIF
import aitom.filter.gaussian as FG
import aitom.image.vol.util as AIVU

from morphsnakes import circle_level_set
from morphsnakes import checkerboard_level_set
from morphsnakes import morphological_chan_vese as ACWE


def sphere_seg(v, sigma, init_level_set=None, out_dir='./output', save_flag=False):
    """
    sphere_seg: membrane segmentation by sphere fitting

    @params:
        v: volume data  sigma: gaussian sigma for denoising
        init_level_set: initial level set for morphsnake

    @return:
        [x, y, z, R]: coordinates and radius of the sphere

    """
    np.random.seed(12345)
    if save_flag:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    # morphsnake
    if init_level_set:
        init = init_level_set
    else:
        circle_set = circle_level_set(v.shape[:2], (80, 80), 60)
        init_mask = np.zeros(v.shape)
        for i in range(init_mask.shape[2]):
            init_mask[:, :, i] = circle_set
        init = checkerboard_level_set(v.shape)
        init = np.multiply(init, init_mask)

    v_im = AIVU.cub_img(v)['im']
    if save_flag:
        plt.imsave(os.path.join(out_dir, 'original.png'), v_im, cmap='gray')
    vg = FG.smooth(v, sigma)
    mask_v = ACWE(image=vg, iterations=25, init_level_set=init)
    mask_im = AIVU.cub_img(mask_v)['im']
    if save_flag:
        plt.imsave(os.path.join(out_dir, 'morphsnake_result.png'), mask_im, cmap='gray')

    # RANSC
    coords = np.array(np.where(mask_v == 1))
    coords = coords.T  # (N,3)

    # robustly fit line only using inlier data with RANSAC algorithm
    xyz = coords

    model_robust, inliers = ransac(xyz, CircleModel3D, min_samples=20, residual_threshold=3, max_trials=50)
    outliers = inliers == False
    # print('inliers_num = ', sum(inliers), 'inliers_num = ', sum(outliers))
    x, y, z, R = model_robust.params
    v_RANSC = np.zeros(mask_v.shape)
    assert len(inliers) == coords.shape[0]
    if save_flag:
        for i in range(len(inliers)):
            if inliers[i]:
                v_RANSC[coords[i][0]][coords[i][1]][coords[i][2]] = 2
            else:
                v_RANSC[coords[i][0]][coords[i][1]][coords[i][2]] = 1
        vim_RANSC = AIVU.cub_img(v_RANSC)['im']
        plt.imsave(os.path.join(out_dir, 'RANSC_result.png'), vim_RANSC, cmap='gray')

        a = FG.smooth(v, sigma)
        a.flags.writeable = True
        color = np.min(a)
        thickness = 1
        center = (x, y, z)
        grid = np.mgrid[[slice(i) for i in a.shape]]
        grid = (grid.T - center).T
        phi1 = R - np.sqrt(np.sum((grid) ** 2, 0))
        phi2 = np.max(R - thickness, 0) - np.sqrt(np.sum((grid) ** 2, 0))
        res = np.int8(phi1 > 0) - np.int8(phi2 > 0)
        a[res == 1] = color
        vim = AIVU.cub_img(a)['im']
        plt.imsave(os.path.join(out_dir, 'result.png'), vim, cmap='gray')
    return model_robust.params


class BaseModel(object):
    def __init__(self):
        self.params = None


class CircleModel3D(BaseModel):

    def estimate(self, data):
        assert data.shape[1] == 3

        def fitfunc(p, coords):
            x0, y0, z0, R = p
            x, y, z = coords.T
            return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

        p0 = [0, 0, 0, 1]

        errfunc = lambda p, x: fitfunc(p, x) - p[3]

        p1, flag = leastsq(errfunc, p0, args=(data,))
        self.params = p1
        return True

    def residuals(self, data):
        assert data.shape[1] == 3

        xc, yc, zc, r = self.params

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        return r - np.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    if n_inliers == 0:
        return np.inf

    nom = 1 - probability
    if nom == 0:
        return np.inf

    inlier_ratio = n_inliers / float(n_samples)
    denom = 1 - inlier_ratio ** min_samples
    if denom == 0:
        return 1
    elif denom == 1:
        return np.inf

    nom = np.log(nom)
    denom = np.log(denom)
    if denom == 0:
        return 0

    return int(np.ceil(nom / denom))


def ransac(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None):
    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    random_state = check_random_state(random_state)

    if min_samples < 0:
        raise ValueError("`min_samples` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if stop_probability < 0 or stop_probability > 1:
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if not isinstance(data, list) and not isinstance(data, tuple):
        data = [data]

    # make sure data is list and not tuple, so it can be modified below
    data = list(data)
    # number of samples
    num_samples = data[0].shape[0]

    for num_trials in range(max_trials):

        # choose random sample set
        samples = []
        random_idxs = random_state.randint(0, num_samples, min_samples)
        for d in data:
            samples.append(d[random_idxs])

        # check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        success = sample_model.estimate(*samples)

        # backwards compatibility
        if success is not None:
            if not success:
                continue

        # check if estimated model is valid
        if is_model_valid is not None \
                and not is_model_valid(sample_model, *samples):
            continue

        sample_model_residuals = np.abs(sample_model.residuals(*data))
        # consensus set / inliers
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = np.sum(sample_model_residuals ** 2)

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(sample_model_inliers)
        if (
            # more inliers
            sample_inlier_num > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (sample_inlier_num == best_inlier_num and sample_model_residuals_sum < best_inlier_residuals_sum)
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            if best_inlier_num >= stop_sample_num or best_inlier_residuals_sum <= stop_residuals_sum or num_trials >=\
                    _dynamic_max_trials(
                    best_inlier_num, num_samples, min_samples, stop_probability):
                break

    # estimate final model using all inliers
    if best_inliers is not None:
        # select inliers for each data array
        for i in range(len(data)):
            data[i] = data[i][best_inliers]
        best_model.estimate(*data)

    return best_model, best_inliers


if __name__ == '__main__':
    path = "/ldap_shared/home/v_zhenxi_zhu/membrane/aitom/membrane.mrc"  # file path
    v = AIF.read_mrc_data(path)
    print(v.shape)
    params = sphere_seg(v=v, sigma=1, out_dir='./output', save_flag=True)
    print('x,y,z,R = ', params)
