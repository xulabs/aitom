"""
functions for loading and saving images

import tomominer.image.io as TIIO
~/ln/tomominer/tomominer/image/io.py
"""

import numpy as N


def format_png_array(m, normalize=True):
    """
    format a 2D array for png saving
    """
    m = N.array(m, dtype=N.float)

    mv = m[N.isfinite(m)]
    if normalize:
        # normalize intensity to 0 to 1
        if mv.max() - mv.min() > 0:
            m = (m - mv.min()) / (mv.max() - mv.min())
        else:
            m = N.zeros(m.shape)
    else:
        assert mv.min() >= 0
        assert mv.max() <= 1

    m = N.ceil(m * 65534)
    m = N.array(m, dtype=N.uint16)

    return m


def save_png(m, name, normalize=True, verbose=False):
    if verbose:
        print('save_png()')
        print('unique values', sorted(set(m.flatten())))

    m = format_png_array(m, normalize=normalize)

    import png  # in pypng package
    png.from_array(m, mode='L;16').save(name)


def save_cub_img(v, name, view_dir=2):
    from . import util as TIVU
    m = TIVU.cub_img(v=v, view_dir=view_dir)['im']
    save_png(m=m, name=name)


def save_image_matplotlib(m, out_file, vmin=None, vmax=None):
    import matplotlib.pyplot as PLT
    import matplotlib.cm as CM

    if vmin is None:
        vmin = m.min()
    if vmax is None:
        vmax = m.max()

    ax_u = PLT.imshow(m, cmap=CM.Greys_r, vmin=vmin, vmax=vmax)
    PLT.axis('off')
    PLT.draw()

    PLT.savefig(out_file, bbox_inches='tight')
    PLT.close("all")
