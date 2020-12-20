"""
a tutorial on using active contour for membrane segmentation
Note: This method implements 2d active contour for each slice separately and then puts the results together.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import aitom.io.file as AIF
import aitom.filter.gaussian as FG
import aitom.image.vol.util as AIVU

from skimage.segmentation import active_contour
from skimage.draw import polygon


def active_contour_slice(v, sigma=3.5, membrane_thickness=5, display_slice=-1, out_dir='./output', save_flag=True):
    """
    active_contour_slice

    @params:
        v: volume data  sigma: gaussian sigma for denoising  membrane_thickness: membrane thickness in voxels
        display_slice: the slice number to be displayed, do nothing if invalid number  out_dir: output dir

    @return:
        mask_v: mask volume
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    mask_v = np.zeros(v.shape)
    print(v.shape)
    vg = FG.smooth(v, sigma)
    for slice_num in range(mask_v.shape[2]):
        save_flag_slice = False
        if slice_num == display_slice and save_flag:
            save_flag_slice = True
        print('processing  slice ', slice_num)
        ori_img = (v[:, :, slice_num]).copy()
        if save_flag_slice:
            plt.imsave(os.path.join(out_dir, 'original.png'), ori_img, cmap='gray')
        img = (vg[:, :, slice_num]).copy()
        if save_flag_slice:
            plt.imsave(os.path.join(out_dir, 'original_smooth.png'), img, cmap='gray')

        # generate init circle
        s = np.linspace(0, 2 * np.pi, 400)
        x = 80 + 60 * np.sin(s)
        y = 80 + 60 * np.cos(s)
        init = np.array([x, y]).T
        # init = create_sphere(20,20,20,10)  # sphere for 3d active contour 
        snake = active_contour(img, init, alpha=0.015, beta=10, gamma=0.001)
        # Note: The default format of the returned coordinates is (x,y) instead of (row,col) in skimage 0.15.x - 0.17.x
        snake[:, [0, 1]] = snake[:, [1, 0]]
        r = snake[:, 0].astype(int)
        c = snake[:, 1].astype(int)

        img2 = img.copy()
        colour = np.min(img2)
        img2[r, c] = colour
        if save_flag_slice:
            plt.imsave(os.path.join(out_dir, 'contour.png'), img2, cmap='gray')

        mask = np.zeros(ori_img.shape)
        rr, cc = polygon(r, c)
        mask[rr, cc] = 2
        mask[r, c] = 1
        for i in range(membrane_thickness):
            mask = contour_shrink(mask)
        mask[mask == 0] = 3
        if save_flag_slice:
            plt.imsave(os.path.join(out_dir, 'mask.png'), mask, cmap='gray')
        mask_v[:, :, slice_num] = mask

    mask_im = AIVU.cub_img(mask_v)['im']

    # better visualize the results
    for i in range(0, mask_im.shape[0], mask_v.shape[0]):
        mask_im[i, :] = 0
    for i in range(0, mask_im.shape[1], mask_v.shape[1]):
        mask_im[:, i] = 0
    if save_flag:
        plt.imsave(os.path.join(out_dir, 'result.png'), mask_im, cmap='gray')

    return mask_v


def create_sphere(cx, cy, cz, r, resolution=360):
    """create sphere with center (cx, cy, cz) and radius r"""
    phi = np.linspace(0, 2 * np.pi, 2 * resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r * np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return np.stack([x, y, z])


def contour_shrink(img):
    result = img.copy()
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if img[i][j] == 1:
                if img[i - 1][j] == 2:
                    result[i - 1][j] = 1
                if img[i + 1][j] == 2:
                    result[i + 1][j] = 1
                if img[i][j - 1] == 2:
                    result[i][j - 1] = 1
                if img[i][j + 1] == 2:
                    result[i][j + 1] = 1

    return result


if __name__ == '__main__':
    path = "/ldap_shared/home/v_zhenxi_zhu/membrane/aitom/membrane.mrc"  # file path
    v = AIF.read_mrc_data(path)
    mask_v = active_contour_slice(v, sigma=3.5, membrane_thickness=5, display_slice=10, out_dir='./output',
                                  save_flag=True)
    unique, counts = np.unique(mask_v, return_counts=True)
    print('mask volume contains(1=membrance voxels, 2=voxels in the membrane, 3=voxels outside the membrane):\n',
          dict(zip(unique, counts)))
