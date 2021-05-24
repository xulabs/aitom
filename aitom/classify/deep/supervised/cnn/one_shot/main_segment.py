import torch
from torch.autograd import Variable

import sys
import os
import cv2
import numpy as np
import denseCRF3D
import argparse
from models import SiameseNetwork
from nets import *
from utilities import preprocess_image, plot_slides, plot_result, plot_result_split, calculate_dice

parser = argparse.ArgumentParser(description='Weakly Supervised One-shot Learning for CECT')

# dataset
parser.add_argument('--dataset', type=str, default='snr10000', help='dataset name: snrINF / snr10000')
parser.add_argument('--data_root', type=str, default='../Data/subtomograms_snr10000/', help='data root folder')
parser.add_argument('--resume', type=str, default='./outputs/2-way-1-shot_scnn_snr10000/model_20.pkl', help='Filename of the checkpoint to resume')

# networks
parser.add_argument('--net', type=str, default='dusescnn', help='network of the model: scnn/dusescnn')
parser.add_argument("--N", type=int, default=2)
parser.add_argument("--K", type=int, default=1)

# training options
parser.add_argument('--n_epochs', type=int, default=40, help='number of epoch')
parser.add_argument('--batchs', type=int, default=1000, help='number of batch per epoch')
parser.add_argument('--batch_size', type=int, default=12, help='training/test batch size')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=4, help='evaluation epochs')

# optimizer
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
args = parser.parse_args()

################################################################################################
class_per_set = args.N
sample_per_class = args.K

n_epochs = args.n_epochs
batches = args.batchs
batch_size = args.batch_size

net = args.net
dataset = args.dataset
data_root = args.data_root

save_dir = 'outputs/{}-way-{}-shot_{}_{}/segment'.format(class_per_set,
                                                         sample_per_class,
                                                         net,
                                                         dataset)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

m = SiameseNetwork(network_type=args.net).cuda()
m.load_state_dict(torch.load(args.resume))
CNN_net = m._modules['net']

dice_all1 = np.zeros((22,2))
dice_all2 = np.zeros((22,2))
for i in range(0, 22):
    print("CLASS  --  ", i)
    vols = np.load(os.path.join(data_root, 'testtom_vol_{}.npy'.format(str(i).zfill(2))))
    segs = np.load(os.path.join(data_root, 'testtom_seg_{}.npy'.format(str(i).zfill(2))))
    vols = preprocess_image(vols)

    dice_c1 = np.zeros((segs.shape[0], 1))
    dice_c2 = np.zeros((segs.shape[0], 1))
    for j in range(segs.shape[0]):
        vol = vols[:,:,j,:,:,:]
        seg = segs[j:j+1,:,:,:]
        _, seg_pred = CNN_net(vol.cuda(), mode='test')

        vol_np = vol[0,0,:,:,:].detach().numpy()
        seg_np = seg[0,:,:,:]
        seg_pred_np = seg_pred[0,0,:,:,:].detach().cpu().numpy()

        # init wrapper object
        dense_crf_param = {}
        dense_crf_param['MaxIterations'] = 4.0
        dense_crf_param['PosW'] = 10.0
        dense_crf_param['PosRStd'] = 10
        dense_crf_param['PosCStd'] = 10
        dense_crf_param['PosZStd'] = 10
        dense_crf_param['BilateralW'] = 30.0
        dense_crf_param['BilateralRStd'] = 100.0
        dense_crf_param['BilateralCStd'] = 100.0
        dense_crf_param['BilateralZStd'] = 100.0
        dense_crf_param['ModalityNum'] = 1
        dense_crf_param['BilateralModsStds'] = (50.0,)

        # run crf and get hard segmentation:
        P = np.repeat(seg_pred_np[:, :, :, np.newaxis], 2, axis=3)
        P[:, :, :, 0] = 1 - P[:, :, :, 1]

        vol_crf = np.uint8((vol_np - vol_np.min()) / (vol_np.max() - vol_np.min()) * 255)
        vol_crf = vol_crf[:,:,:,np.newaxis]
        seg_pred_crf_np = denseCRF3D.densecrf3d(vol_crf, P, dense_crf_param)

        # get raw hard segmentation
        seg_pred_np[seg_pred_np <= 0.5] = 0
        seg_pred_np[seg_pred_np > 0.5] = 1

        # save the image and segmentation visualizations
        s1 = plot_slides(vol_np)
        cv2.imwrite(os.path.join(save_dir, 'c{}_{}_vol.png'.format(i, j)), s1)
        s1 = plot_slides(seg_np)
        cv2.imwrite(os.path.join(save_dir, 'c{}_{}_seg.png'.format(i, j)), s1)
        s1 = plot_slides(seg_pred_np)
        cv2.imwrite(os.path.join(save_dir, 'c{}_{}_seg_pred.png'.format(i, j)), s1)
        s1 = plot_slides(seg_pred_crf_np)
        cv2.imwrite(os.path.join(save_dir, 'c{}_{}_seg_pred_crf.png'.format(i, j)), s1)

        # compute the dice
        dice_c1[j] = calculate_dice(seg_np, seg_pred_crf_np)
        dice_c2[j] = calculate_dice(seg_np, seg_pred_np)

    m_dice_c1 = np.mean(dice_c1)
    std_dice_c1 = np.std(dice_c1)
    dice_all1[i, 0] = m_dice_c1
    dice_all1[i, 1] = std_dice_c1

    m_dice_c2 = np.mean(dice_c2)
    std_dice_c2 = np.std(dice_c2)
    dice_all2[i, 0] = m_dice_c2
    dice_all2[i, 1] = std_dice_c2

print(dice_all1)
print(dice_all2)

np.savetxt(os.path.join(save_dir, "dice_all1.txt"), dice_all1)
np.savetxt(os.path.join(save_dir, "dice_all2.txt"), dice_all2)
