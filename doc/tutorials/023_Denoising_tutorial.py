import os, sys
print(os.getcwd())
os.chdir("../..") # Depends on your current dir
print(os.getcwd())
sys.path.append(".")
import aitom.filter.gaussian as G
import numpy as np
import aitom.io.mrcfile_proxy as mrcfile_proxy
import aitom.io.file as io_file
import time
import matplotlib.pyplot as plt
import aitom.filter.anistropic_diffusion.fastaniso as ft
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import aitom.image.vol.util as GV
import mrcfile

save_dir = './Denoising_Result'

def write_with_spacing(data, voxel_spacing, path, overwrite=False):
    # only for 3D array
    assert data.ndim == 3
    data = data.astype(np.float32)
    # transpose data according to tomominer.image.vol.eman2_util.em2numpy
    data = data.transpose([2, 1, 0])
    with mrcfile.new(path, overwrite=overwrite) as m:
        m.set_data(data)
        m.voxel_size = voxel_spacing

def g_denoising(G_type, a, voxel_spacing, name, gaussian_sigma, save_flag=False):
    b_time = time.time()

    if G_type == 1:
        a = G.smooth(a, gaussian_sigma)
    elif G_type == 2:
        a = G.dog_smooth(a, gaussian_sigma)
    elif G_type == 3:
        a = G.dog_smooth__large_map(a, gaussian_sigma)

    end_time = time.time()
    print('Gaussian de-noise takes', end_time - b_time, 's', ' sigma=', gaussian_sigma)

    if save_flag:
        img = (a[:, :, int(a.shape[2] / 2)]).copy()
        # TODO: Change the image and tomogram saving path
        img_path = save_dir + '/Gaussian/' + str(name) + '_G=' + \
                   str(gaussian_sigma) + '_type=' + str(G_type) + '.png'
        plt.imsave(img_path, img, cmap='gray')

        mrc_path = save_dir + '/Gaussian/' + str(name) + '_G=' + \
                   str(gaussian_sigma) + '_type=' + str(G_type) + '.mrc'
        #io_file.put_mrc_data(a, mrc_path)
        write_with_spacing(a, voxel_spacing, mrc_path)

        return img


def bandpass_denoising(a, voxel_spacing, name, save_flag=False):
    b_time = time.time()
    grid = GV.grid_displacement_to_center(a.shape, GV.fft_mid_co(a.shape))
    rad = GV.grid_distance_to_center(grid)
    rad = np.round(rad).astype(np.int)

    # create a mask that only center frequencies components will be left
    curve = np.zeros(rad.shape)
    # TODO: change the curve value as desired
    curve[int(rad.shape[0] / 8) * 3: int(rad.shape[0] / 8) * 5, int(rad.shape[1] / 8) * 3: int(rad.shape[1] / 8) * 5,
    int(rad.shape[2] / 8) * 3: int(rad.shape[2] / 8) * 5] = 1

    #perform FFT and filter the data with the mask and then transform the filtered data back
    vf = ifftn(ifftshift((fftshift(fftn(a)) * curve)))
    vf = np.real(vf)
    
    end_time = time.time()
    print('Bandpass de-noise takes', end_time - b_time, 's')

    if save_flag:
        img = (vf[:, :, int(vf.shape[2] / 2)]).copy()
        # TODO: Change the image and tomogram saving path
        img_path = save_dir + '/Bandpass/' + str(name) + '_BP.png'
        plt.imsave(img_path, img, cmap='gray')

        mrc_path = save_dir + '/Bandpass/' + str(name) + '_BP.mrc'
        #io_file.put_mrc_data(vf, mrc_path)
        write_with_spacing(vf, voxel_spacing, mrc_path)

        return img


def anistropic_diffusion(a, voxel_spacing, niter, kappa, gamma, step, option, ploton, name, save_flag=False):
    b_time = time.time()

    #see http://pastebin.com/sBsPX4Y7 for details
    af = ft.anisodiff3(stack=a, niter=niter, kappa=kappa, gamma=gamma, step=step, option=option, ploton=ploton)

    end_time = time.time()
    print('Anistropic Diffusion de-noise takes', end_time - b_time, 's')

    if save_flag:
        af_fimg = (af[:, :, int(af.shape[2] / 2)]).copy()
        # TODO: Change the image and tomogram saving path
        af_fimg_path = save_dir + '/Anistropic_Diffusion/' + str(name) + \
                       '_AD_i=' + str(niter) + '_k=' + str(kappa) + '_g=' + str(gamma) + '.png'
        plt.imsave(af_fimg_path, af_fimg, cmap='gray')

        mrc_path = save_dir + '/Anistropic_Diffusion/' + str(name) + \
                       '_AD_i=' + str(niter) + '_k=' + str(kappa) + '_g=' + str(gamma) + '.mrc'
        #io_file.put_mrc_data(af, mrc_path)
        write_with_spacing(af, voxel_spacing, mrc_path)

        return af_fimg


def diff_compare(oimg, fimg, type):
    dimg = fimg - oimg
    dmax = np.amax(dimg)
    dimg = (dimg / dmax) * 255
    # TODO: Change the image saving path
    img_path = save_dir + '/Difference_' + str(type) + '.png'
    plt.imsave(img_path, dimg, cmap='gray')


if __name__ == "__main__":
    # TODO: Change the data path, name and Gussian denoise type
    path = '../local_scratch/aitom_demo_single_particle_tomogram.mrc'
    name = 'aitom_demo_single_particle_tomogram'
    G_type = 1

    # read the volume data as numpy array
    original = mrcfile_proxy.read_data(path)

    mrc_header = io_file.read_mrc_header(path)
    voxel_spacing = mrc_header['MRC']['xlen'] / mrc_header['MRC']['nx']
    print(voxel_spacing)

    # save a slice of original tomogram
    oimg = (original[:, :, int(original.shape[2] / 2)]).copy()
    # TODO: Change the orginal image saving directory
    img_path = save_dir + '/Original.png'
    plt.imsave(img_path, oimg, cmap='gray')

    # perform three diffrent kind of denoising
    # The difference comparison between original tomogram and filtered tomogram is commented out
    g_fimg = g_denoising(G_type, a=original, voxel_spacing=voxel_spacing, name=name, gaussian_sigma=2.5, save_flag=True)
    # diff_compare(oimg, g_fimg, "Gaussian")

    bp_fimg = bandpass_denoising(a=original, voxel_spacing=voxel_spacing, name=name, save_flag=True)
    # diff_compare(oimg, bp_fimg, "BP")

    af_fimg = anistropic_diffusion(a=original, voxel_spacing=voxel_spacing, niter=70, kappa=100, gamma=0.25, 
            step=(4., 4., 4.), option=1, ploton=True, name=name, save_flag=True)
    # diff_compare(oimg, af_fimg, "AF")
