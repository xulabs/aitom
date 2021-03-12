"""
Tutorial on autoencoder

References:
https://arxiv.org/abs/1706.04970
https://github.com/xulabs/projects/tree/master/autoencoder

Requirements:
- Keras need to be 2.1.x for autoencoder
- run on GPU
"""

import argparse
import os
import time
from os.path import join as op_join
import aitom.io.file as IF
import aitom.classify.deep.unsupervised.autoencoder.autoencoder as AE
import aitom.classify.deep.unsupervised.autoencoder.seg_src as SEG


def parse_args():
    parser = argparse.ArgumentParser(description='autoencoder')
    parser.add_argument('--data', dest='data', default=None, type=str)
    parser.add_argument('--clu-num', dest='cluster_number', default=None, type=int)
    parser.add_argument('--output-dir', dest='output_dir', default='./tmp/autoencoder', type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def main(args):
    """
    Step1:Prepare input dataset

    You can download the example dataset from
    https://cmu.app.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp/file/509296892992 into your present working directory.
    Note: This is not the dataset used in the paper, which will be added in the future.

    Here's four parameters and dataset format.
    1.A python pickle data file of CECT small subvolumes, this data file should be prepared as follows:
        d is the small subvolume data file.
        d is a dictionary consists 'v_siz' and 'vs'.
        d['v_siz'] is an numpy.ndarray specifying the shape of the small subvolume.
        For example, d['v_siz'] = array([32,32,32]).
        d['vs'] is a dictionary with keys of uuids specifying each small subvolume.
        d['vs'][an example uuid] is a dictionary consists 'center', 'id', and 'v'.
        d['vs'][an example uuid]['center'] is the center of the small subvolume in the tomogram.
        For example, d['vs'][an example uuid]['center'] = [110,407,200].
        d['vs'][an example uuid]['id'] is the specific uuid.
        d['vs'][an example uuid]['v'] are voxel values of the small subvolume, which is an numpy.ndarray of
        shape d['v_siz'].
    2.A tomogram file in .rec format, which is only required when performing pose normalization.
    3.Whether the optional pose normalization step should be applied. Input should be True or False.
    4.The number of clusters. This should be an positive integer such as 100.
    """

    """
    Step2 Train the auto encoder. Given the example dataset, you can use parameters1 or parameters2.
    """
    s_time = time.time()
    # Note: these two datasets are missing for now
    # parameters1 = ["example/subvolumes_example_2.pickle", "None", "False", "4"]
    # parameters2 = ["example/subvolumes_example_1.pickle", "example/tomogram.rec", "True", "100"]

    # This example dataset is generated in the particle picking tutorial
    single_particle_param = ['data/demo_single_particle_subvolumes.pickle', 'None', "False", 4]
    out_dir = args.output_dir
    if args.data:
        single_particle_param[0] = args.data
    if args.cluster_number:
        single_particle_param[3] = args.cluster_number

    # demo dataset format is different, which has 2 kinds of particle and their templates
    multiple_particles_params = ['data/aitom_demo_subtomograms.pickle', 'None', "False", 4]

    parameters_demo = single_particle_param  # choose one of the above
    d = IF.pickle_load(parameters_demo[0])  # pickle data file of CECT small subvolumes

    if parameters_demo == multiple_particles_params:  # TODO add multiparticle tutorial
        # choose one particle and convert the data format
        # d = d['5T2C_data']
        # from aitom.classify.deep.unsupervised.autoencoder.autoencoder_util import subtomograms_to_subvolumes
        # d = subtomograms_to_subvolumes(d)
        pass

    # A tomogram file in .rec format, which can be None when pose normalization is not required
    img_org_file = parameters_demo[1]
    # Whether the optional pose normalization step should be applied: True or False
    pose = eval(parameters_demo[2])
    # The number of clusters
    clus_num = int(parameters_demo[3])

    AE.encoder_simple_conv_test(d=d, pose=pose, img_org_file=img_org_file, out_dir=out_dir, clus_num=clus_num)
    AE.kmeans_centers_plot(op_join(out_dir, 'clus-center'))

    """
    Step 3. Manual selection of small subvolume clusters
    
    Autoencoder3D training step will have two output folders.
    'model' directory saved the trained models
    'clus-center' directory for the resulting clusters.
     
    There should be two pickle files in 'clus-center'. 
    'kmeans.pickle' stores the uuids for each cluster.
    'ccents.pickle' stores the decoded cluster centers.
    
    The 'fig' folder under 'clus-center' directory contains the 2D slices of decoded cluster center. 
    User can use the figures as guide for manual selection.
      
    Manual selection clues are provided in the folder 'fig' under 'clus-center'.
    Each picture is a 2D slices presentation of a decoded small subvolume cluster center.
    The picture name such as '035--47.png' refers to cluster 35 which consists 47 small subvolumes.
    """

    """
    Step 4. Optional Encoder-decoder Semantic Segmentation 3D network training.
    Based on the manual selection results, Encoder-decoder Semantic Segmentation 3D (EDSS3D) network can be trained
    and applied for another tomogram dataset.
    """

    # sel_clus is the selected clusters for segmentation, which can be multiple classes.
    sel_clus = {0: [3, 21, 28, 34, 38, 39, 43, 62, 63, 81, 86, 88]}
    # sel_clus = {1: [3, 21, 28, 34, 38, 39, 43, 62, 63, 81, 86, 88],
    #             2: [15, 25, 29, 33, 35, 66, 79, 90, 92, 98]}
    # import numpy as np
    # sel_clus = {0: np.arange(100)}
    data_dir = out_dir
    # here's the name of pickle data file of CECT small subvolumes
    data_file = op_join(data_dir, parameters_demo[0])

    with open(data_file, 'rb') as f:
        d = SEG.pickle.load(f, encoding='iso-8859-1')

    SEG.decode_all_images(d, data_dir)

    # The following files come from the previous Autoencoder3D results
    with open(op_join(data_dir, 'clus-center', 'kmeans.pickle'), 'rb') as f:
        km = SEG.pickle.load(f, encoding='iso-8859-1')
    with open(op_join(data_dir, 'clus-center', 'ccents.pickle'), 'rb') as f:
        cc = SEG.pickle.load(f, encoding='iso-8859-1')
    with open(op_join(data_dir, 'decoded', 'decoded.pickle'), 'rb') as f:
        vs_dec = SEG.pickle.load(f, encoding='iso-8859-1')

    print('sel_clus', len(sel_clus))
    print('km', len(km))
    vs_lbl = SEG.image_label_prepare(sel_clus, km)
    print('vs_lbl', len(vs_lbl))
    # print(vs_lbl)
    for key in vs_lbl:
        vs_lbl[key] = 1
    print('vs_lbl', len(vs_lbl))
    # print(vs_lbl)

    # iso_value is the mask threshold for segmentation
    vs_seg = SEG.train_label_prepare(vs_lbl=vs_lbl, vs_dec=vs_dec, iso_value=0.5)
    print('vs_seg', len(vs_seg))
    model_dir = op_join(data_dir, 'model-seg')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_checkpoint_file = op_join(model_dir, 'model-seg--weights--best.h5')
    model_file = op_join(model_dir, 'model-seg.h5')

    if os.path.isfile(model_file):
        print('use existing', model_file)
        import keras.models as KM
        model = KM.load_model(model_file)
    else:
        model = SEG.train_validate__reshape(vs_lbl=vs_lbl, vs=d['vs'], vs_seg=vs_seg, model_file=model_file,
                                            model_checkpoint_file=model_checkpoint_file)
        model.save(model_file)

    # Segmentation prediction on new data
    data_dir = out_dir  # This should be the new data for prediction
    data_file = op_join(data_dir, parameters_demo[0])
    with open(data_file, 'rb') as f:
        d = SEG.pickle.load(f, encoding='iso-8859-1')

    prediction_dir = op_join(data_dir, 'prediction')
    if not os.path.isdir(prediction_dir):
        os.makedirs(prediction_dir)
    vs_p = SEG.predict__reshape(model, vs={_: d['vs'][_]['v'] for _ in vs_seg})

    with open(op_join(prediction_dir, 'vs_p.pickle'), 'wb') as f:
        SEG.pickle.dump(vs_p, f, protocol=-1)
    print(time.time() - s_time, 's')


if __name__ == '__main__':
    arguments = parse_args()
    print(arguments)
    main(arguments)
