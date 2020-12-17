import sys
import numpy as N
import os

op = {'mrcfile': '../IOfile/tomo/mrc/tomo_SNR04.mrc',
      'pngdir': '../IOfile/tomo/png/tomo2/SNR04/',
      'pngname': 'SNR04',
      'view_dir': 1}

# view_dir =  0, 1, 2


def cub_img(v, view_dir=2):
    """convert a 3D cube to a 2D image of slices"""
    if view_dir == 0:
        vt = N.transpose(v, [1, 2, 0])
    elif view_dir == 1:
        vt = N.transpose(v, [2, 0, 1])
    elif view_dir == 2:
        vt = v

    row_num = vt.shape[0] + 1
    col_num = vt.shape[1] + 1
    slide_num = vt.shape[2]
    disp_len = int(N.ceil(N.sqrt(slide_num)))

    slide_count = 0
    im = N.zeros((row_num * disp_len, col_num * disp_len)) + float('nan')
    for i in range(disp_len):
        for j in range(disp_len):
            im[(i * row_num): ((i + 1) * row_num - 1), (j * col_num): ((j + 1) * col_num - 1)] = vt[:, :, slide_count]
            slide_count += 1

            if slide_count >= slide_num:
                break

        if slide_count >= slide_num:
            break

    im_v = im[N.isfinite(im)]

    if im_v.max() > im_v.min():
        im = (im - im_v.min()) / (im_v.max() - im_v.min())

    return {'im': im, 'vt': vt}


def format_png_array(m, normalize=True):
    """format a 2D array for png saving"""
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
    # print("max, min:", m.max(), m.min())
    # print("=========")
    # print(m.shape)
    # print("=========")
    return m


def save_png(m, name, normalize=True, verbose=False):
    if verbose:
        print('save_png()')
        print('unique values', sorted(set(m.flatten())))

    m = format_png_array(m, normalize=normalize)

    import png  # in pypng package
    png.from_array(m, mode='L;16').save(name)


def mrc2singlepic(op):
    from . import iomap as IM
    data = IM.readMrcMap(op['mrcfile'])
    if op['view_dir'] == 0:
        data = N.transpose(data, [1, 2, 0])
    elif op['view_dir'] == 1:
        data = N.transpose(data, [2, 0, 1])
    elif op['view_dir'] == 2:
        data = data

    shape = data.shape
    for j in range(shape[0]):
        d = data[j]
        name = op['pngdir'] + op['pngname'] + '_' + str(j) + '.png'
        if not os.path.exists(op['pngdir']):
            os.makedirs(op['pngdir'])
        save_png(d, name)
        print('save' + name)


if __name__ == '__main__':
    try:
        mrc2singlepic(sys.argv[1])
    except:
        mrc2singlepic(op)
