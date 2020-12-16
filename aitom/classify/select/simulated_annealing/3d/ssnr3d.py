from __future__ import division
import copy as C
import numpy as N
from numpy.fft import fftn, ifftn, fftshift, ifftshift


class SSNR3D:
    """
    input: images is a dictionary of 3D images indexed by keys;
    masks is a dictionary of 3D images of binary values
    """
    def __init__(self, images, masks, band_width_radius=1.0):
        im_f = dict()
        for k in images:
            im = images[k]
            im = fftshift(fftn(im))
            im_f[k] = im

        # fft transformed images
        self.im_f = im_f
        # masks
        self.ms = masks
        self.ks = set()
        self.img_siz = im_f[k].shape
        self.set_fft_mid_co()
        self.set_rad()
        self.band_width_radius = band_width_radius

    def set_img_set(self, ks):
        for k in ks:
            assert k in self.im_f
        self.ks = C.deepcopy(ks)
        self.ks = set(self.ks)
        self.update_summary_statistics()

    def add_to_set(self, k):
        assert k in self.im_f
        assert k not in self.ks
        self.ks.add(k)
        self.sum_v += self.im_f[k]
        self.prod_sum_v += self.im_f[k] * N.conj(self.im_f[k])
        self.mask_sum_v += self.ms[k]

    def remove_from_set(self, k):
        assert k in self.im_f
        assert k in self.ks
        self.ks.remove(k)
        self.sum_v -= self.im_f[k]
        self.prod_sum_v -= self.im_f[k] * N.conj(self.im_f[k])
        self.mask_sum_v -= self.ms[k]

    def update_summary_statistics(self):
        sum_v = N.zeros(self.img_siz, dtype=N.complex)
        for k in self.ks:
            sum_v += self.im_f[k]

        prod_sum_v = N.zeros(self.img_siz, dtype=N.complex)
        for k in self.ks:
            prod_sum_v += self.im_f[k] * N.conj(self.im_f[k])

        mask_sum_v = N.zeros(self.img_siz, dtype=float)
        for k in self.ks:
            mask_sum_v += self.ms[k]

        self.sum_v = sum_v
        self.prod_sum_v = prod_sum_v
        self.mask_sum_v = mask_sum_v

    def set_fft_mid_co(self):
        siz = self.img_siz
        assert (N.all(N.mod(siz, 1) == 0))
        assert (N.all(N.array(siz) > 0))

        mid_co = N.zeros(len(siz))

        # according to following code that uses numpy.fft.fftshift()
        for i in range(len(mid_co)):
            m = siz[i]
            mid_co[i] = N.floor(m / 2)
        self.mid_co = mid_co

    def grid_displacement_to_center(self):
        size = N.array(self.img_siz, dtype=N.float)
        assert size.ndim == 1

        grid = N.mgrid[0:size[0], 0:size[1], 0:size[2]]

        for dim in range(3):
            grid[dim, :, :] -= self.mid_co[dim]

        return grid

    def grid_distance_sq_to_center(self, grid):
        dist_sq = N.zeros(grid.shape[1:])
        if grid.ndim == 4:
            for dim in range(3):
                dist_sq += N.squeeze(grid[dim, :, :, :]) ** 2
        elif grid.ndim == 3:
            for dim in range(2):
                dist_sq += N.squeeze(grid[dim, :, :]) ** 2
        else:
            assert False

        return dist_sq

    def grid_distance_to_center(self, grid):
        dist_sq = self.grid_distance_sq_to_center(grid)
        return N.sqrt(dist_sq)

    # get a volume containing radius
    def set_rad(self):
        grid = self.grid_displacement_to_center()
        self.rad = self.grid_distance_to_center(grid)

    # get index within certain frequency band
    def rad_ind(self, r):
        return abs(self.rad - r) <= self.band_width_radius

    def get_ssnr(self):
        ind = self.mask_sum_v > 2
        avg = N.zeros(self.sum_v.shape, dtype=N.complex) + N.nan
        avg[ind] = self.sum_v[ind] / self.mask_sum_v[ind]

        avg_abs_sq = N.zeros(self.sum_v.shape, dtype=N.complex) + N.nan
        avg_abs_sq[ind] = avg[ind] * N.conj(avg[ind])

        var = N.zeros(self.sum_v.shape, dtype=N.complex) + N.nan
        var[ind] = (self.prod_sum_v[ind] -
                    self.mask_sum_v[ind] * avg_abs_sq[ind]) / (self.mask_sum_v[ind] - 1)
        var = N.real(var)

        vol_rad = int(N.floor(N.min(self.img_siz) / 2.0) + 1)
        # this is the SSNR of the AVERAGE image
        ssnr = N.zeros(vol_rad) + N.nan

        # the interpolation can also be performed using scipy.ndimage.interpolation.map_coordinates()
        for r in range(vol_rad):
            # in order to use it as an index or mask, must convert to a bool array, not integer array!!!!
            ind = self.rad_ind(r=r)
            ind[N.logical_not(N.isfinite(avg))] = False
            ind[N.logical_not(N.isfinite(var))] = False

            if var[ind].sum() > 0:
                ssnr_t = (self.mask_sum_v[ind] * avg_abs_sq[ind]).sum() / var[ind].sum()
            else:
                ssnr_t = 0.0
            ssnr[r] = N.real(ssnr_t)

        assert N.all(N.isfinite(ssnr))
        return ssnr

    def get_fsc(self):
        ssnr = self.get_ssnr()
        fsc = ssnr / (2.0 + ssnr)
        return fsc


    def get_fsc_sum(self):
        """
        this is the objective function to be minimized
        """
        return self.get_fsc().sum()
