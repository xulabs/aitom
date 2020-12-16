"""
a tutorial on using subtomogram averaging or classification
"""

import os
import pickle
import shutil
import uuid

import numpy as N

import aitom.io.db.lsm_db as TIDL
import aitom.io.file as AIF


def cub_img(v, view_dir=2):
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
            im[(i * row_num):((i + 1) * row_num - 1),
               (j * col_num):((j + 1) * col_num - 1)] = vt[:, :, slide_count]
            slide_count += 1

            if slide_count >= slide_num:
                break

        if slide_count >= slide_num:
            break

    im_v = im[N.isfinite(im)]

    if im_v.max() > im_v.min():
        im = (im - im_v.min()) / (im_v.max() - im_v.min())

    return {'im': im, 'vt': vt}


def save_png(m, name, normalize=True, verbose=False):
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

    # in pypng package
    import png
    png.from_array(m, 'L').save(name)


def pickle_load(path):
    with open(path, 'rb') as f:
        o = pickle.load(f)
    return o


if __name__ == '__main__':
    # generate image.db and data.pickle from existing pickle file
    # Download from: https://cmu.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp
    path = './aitom_demo_subtomograms.pickle'
    with open(path, 'rb') as f:
        # 'data' is a dict containing several different subtomograms.
        # 'data['5T2C_data']' is a list containing 100 three-dimensional arrays (100 subtomograms).
        data = pickle.load(f, encoding='iso-8859-1')
    # for key in data:
    #     print(key)  # 1KP8_data

    # choose average only or classify
    classify = False
    average = True
    subtom = []
    if classify:
        # test with 2 classes
        subtom = data['5T2C_data'] + data['1KP8_data']
    elif average:
        subtom = data['5T2C_data']

    assert len(subtom) > 0
    print(len(subtom), subtom[0].shape)
    # # 32x32x32 volume
    # v = data['5T2C_data'][0]
    # v = v.astype(N.float)

    # test dir
    test_dir = './tmp/cls-test'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    dj_file = os.path.join(test_dir, 'data.pickle')
    img_db_file = os.path.join(test_dir, 'image.db')

    # number of each class
    v_num = 100
    v_dim_siz = 32
    wedge_angle = 30
    mask_id = str(uuid.uuid4())
    dj = []
    if classify:
        class_num = 2
    else:
        class_num = 1
    for model_id in range(class_num):
        for v_i in range(v_num):
            ang_t = [_ for _ in N.random.random(3) * (N.pi * 2)]
            # loc_t = TGA.random_translation(size=[v_dim_siz]*3, proportion=0.2)
            loc_t = [0.0, 0.0, 0.0]
            v_id = str(uuid.uuid4())
            dj.append({
                'subtomogram': v_id,
                'mask': mask_id,
                'angle': ang_t,
                'loc': loc_t,
                'model_id': model_id
            })
    AIF.pickle_dump(dj, dj_file)

    sim_op = {
        'model': {
            'missing_wedge_angle': wedge_angle,
            'titlt_angle_step': 1,
            'SNR': 1000,
            'band_pass_filter': False,
            'use_proj_mask': False},
        'ctf': {
            'pix_size': 1.0,
            'Dz': -5.0,
            'voltage': 300,
            'Cs': 2.0,
            'sigma': 0.4}}

    img_db = TIDL.LSM(img_db_file)
    index = 0
    for d in dj:
        img_db[d['subtomogram']] = subtom[index].astype(N.float)
        # print(img_db[d['subtomogram']].shape)
        index = index + 1

    import aitom.image.vol.wedge.util as TIVWU
    img_db[mask_id] = TIVWU.wedge_mask(size=[v_dim_siz] * 3, ang1=wedge_angle)
    print('file generation complete')

    out_dir = os.path.join(test_dir, 'out')
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    from aitom.classify.align.simple_iterative.classify import randomize_orientation
    from aitom.classify.align.simple_iterative.classify import export_avgs
    if classify: # classification and averaging
        import aitom.classify.align.simple_iterative.classify as clas
        class_num = 2
        op = dict()
        op['option'] = {'pass_num': 20} # the number of iterations
        op['data_checkpoint'] = os.path.join(out_dir, 'djs.pickle')
        op['dim_reduction'] = {}
        op['dim_reduction']['pca'] = {
            'n_dims': 50,
            'n_iter': 10,
            'checkpoint': os.path.join(out_dir, 'pca.pickle')
        }
        op['clustering'] = {
            'kmeans_k': class_num,
            'checkpoint': os.path.join(out_dir, 'clustering.pickle')
        }
        op['average'] = {}
        op['average']['mask_count_threshold'] = 2
        op['average']['checkpoint'] = os.path.join(out_dir, 'avgs.pickle')

        dj = AIF.pickle_load(os.path.join(test_dir, 'data.pickle'))
        img_db = TIDL.LSM(os.path.join(test_dir, 'image.db'), readonly=True)

        randomize_orientation(dj)
        clas.classify(dj_init=dj, img_db=img_db, op=op)

        export_avgs(AIF.pickle_load(os.path.join(out_dir, 'avgs.pickle')),
                    out_dir=os.path.join(out_dir, 'avgs-export'))
        print('classification complete')
    else: # averaging only
        import aitom.average.align.simple_iterative.average as avg
        op = dict()
        op['option'] = {'pass_num': 20} # the number of iterations
        op['data_checkpoint'] = os.path.join(out_dir, 'djs.pickle')
        op['average'] = {}
        op['average']['mask_count_threshold'] = 2
        op['average']['checkpoint'] = os.path.join(out_dir, 'avgs.pickle')

        dj = AIF.pickle_load(os.path.join(test_dir, 'data.pickle'))
        img_db = TIDL.LSM(os.path.join(test_dir, 'image.db'), readonly=True)

        randomize_orientation(dj)
        avg.average(dj_init=dj, img_db=img_db, op=op)

        export_avgs(AIF.pickle_load(os.path.join(out_dir, 'avgs.pickle')),
                    out_dir=os.path.join(out_dir, 'avgs-export'))
        print('averaging done')

    # visualization
    avgs = pickle_load('./tmp/cls-test/out/avgs.pickle')
    out_dir = os.path.join(test_dir, 'image')
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for i in avgs.keys():
        v = avgs[i]['v']
        file_name = str(avgs[i]['pass_i']) + '_' + str(i) + '.png'
        save_png(cub_img(v)['im'], os.path.join(out_dir, file_name))
    print('images saved')
