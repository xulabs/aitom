'''
utility functions for images
'''

import numpy as N

def roll(v, s0, s1, s2):
    if (s0 != 0):
        v0 = N.roll(v, s0, axis=0)
    else:
        v0 = v
    if (s1 != 0):
        v1 = N.roll(v0, s1, axis=1)
    else:
        v1 = v0
    if (s2 != 0):
        v2 = N.roll(v1, s2, axis=2)
    else:
        v2 = v1
    return v2
import scipy.ndimage.interpolation as inter

def fft_mid_co(siz):
    assert all((N.mod(siz, 1) == 0))
    assert all((N.array(siz) > 0))
    mid_co = N.zeros(len(siz))
    for i in range(len(mid_co)):
        m = siz[i]
        mid_co[i] = N.floor((m / 2))
    return mid_co

def grid_displacement_to_center(size, mid_co=None):
    size = N.array(size, dtype=N.float)
    assert (size.ndim == 1)
    if (mid_co is None):
        mid_co = ((N.array(size) - 1) / 2)
    if (size.size == 3):
        grid = N.mgrid[0:size[0], 0:size[1], 0:size[2]]
        for dim in range(3):
            grid[dim, :, :, :] -= mid_co[dim]
    elif (size.size == 2):
        grid = N.mgrid[0:size[0], 0:size[1]]
        for dim in range(2):
            grid[dim, :, :] -= mid_co[dim]
    else:
        assert False
    return grid

def grid_distance_sq_to_center(grid):
    dist_sq = N.zeros(grid.shape[1:])
    if (grid.ndim == 4):
        for dim in range(3):
            dist_sq += (N.squeeze(grid[dim, :, :, :]) ** 2)
    elif (grid.ndim == 3):
        for dim in range(2):
            dist_sq += (N.squeeze(grid[dim, :, :]) ** 2)
    else:
        assert False
    return dist_sq

def grid_distance_to_center(grid):
    dist_sq = grid_distance_sq_to_center(grid)
    return N.sqrt(dist_sq)

# display an image
def dsp_img(v, new_figure=True):

    import matplotlib.pyplot as plt

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = plt


    import matplotlib.cm as cm
    
    ax_u = ax.imshow(  v, cmap = cm.Greys_r )
    ax.axis('off') # clear x- and y-axes

    plt.pause(0.001)        # calling pause will display the figure without blocking the program, see segmentation.active_contour.morphsnakes.evolve_visual


