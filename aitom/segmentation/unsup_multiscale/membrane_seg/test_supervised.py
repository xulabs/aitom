import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm   

from networks.net_factory import net_factory
from networks.backboned_unet.unet import Unet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='data/', help='Root directory containing the dataset')
parser.add_argument('--exp', type=str, default='test', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=10, help='labeled data')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--pretrain', type=int,  default=0, help='use pretrained encoder')

sample_save_count = 0

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    if np.sum(pred) == 0 and np.sum(gt) == 0:
        # Both empty — perfect match
        return 1.0, 1.0, 0.0, 0.0
    elif np.sum(pred) == 0 or np.sum(gt) == 0:
        # One is empty, the other is not — worst case
        return 0.0, 0.0, 100, 100
    else:
        dice = metric.dc(pred, gt)
        jc = metric.jc(pred, gt)
        hd95 = metric.hd95(pred, gt)
        asd = metric.asd(pred, gt)
        return dice, jc, hd95, asd

def test_single_volume(case, net, test_save_path, FLAGS):
    global sample_save_count
    h5f = h5py.File(os.path.join(FLAGS.root_path, f"data/{case}.h5"), 'r')
    image = h5f['image'][:]  # shape: (H, W)
    image = (image - image.mean()) / (image.std() + 1e-8)
    label = h5f['label'][:]  # shape: (H, W)
    
    H, W = image.shape
    patch_size = 512
    prediction = np.zeros((H, W), dtype=np.uint8)
    
    net.eval()
    with torch.no_grad():
        for i in range(2):  # top-bottom
            for j in range(2):  # left-right
                h_start = i * patch_size
                h_end = (i + 1) * patch_size
                w_start = j * patch_size
                w_end = (j + 1) * patch_size
                
                img_patch = image[h_start:h_end, w_start:w_end]
                input_tensor = torch.from_numpy(img_patch).unsqueeze(0).unsqueeze(0).float().cuda()  # (1, 1, 256, 256)
                if FLAGS.pretrain:
                    input_tensor = input_tensor.repeat(1, 3, 1, 1)
                output = net(input_tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                pred_patch = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()
                
                prediction[h_start:h_end, w_start:w_end] = pred_patch

    # Metrics
    class_metrics = []
    for class_idx in range(1, FLAGS.num_classes):
        pred_mask = (prediction == class_idx)
        gt_mask = (label == class_idx)
        if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
            metrics = (1.0, 1.0, 0.0, 0.0)
        elif np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
            metrics = (0.0, 0.0, 100.0, 100.0)
        else:
            metrics = calculate_metric_percase(pred_mask, gt_mask)
        class_metrics.append(metrics)

    # Save results
    if np.sum(gt_mask) > 0 and sample_save_count < 30:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.SetSpacing((1, 1, 1))
        pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
        pred_itk.SetSpacing((1, 1, 1))
        label_itk = sitk.GetImageFromArray(label.astype(np.uint8))
        label_itk.SetSpacing((1, 1, 1))
    
        sitk.WriteImage(pred_itk, os.path.join(test_save_path, f"{case}_pred.nii.gz"))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case}_img.nii.gz"))
        sitk.WriteImage(label_itk, os.path.join(test_save_path, f"{case}_gt.nii.gz"))
        sample_save_count += 1

    return class_metrics

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "../model/supervised/{}_{}_labeled".format(FLAGS.exp, FLAGS.labelnum)
    test_save_path = "../model/supervised/{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    if FLAGS.pretrain:
        net = Unet('resnet50', True, classes=FLAGS.num_classes).cuda()
    else:
        net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path)['net'])

    print("init weight from {}".format(save_model_path))
    net.eval()

    groundtruth_total = 0.0

    
    score_samples = []
    
    for case in tqdm(image_list):
        grountruth_metric = test_single_volume(case, net, test_save_path, FLAGS)
        score_samples.append(np.asarray(grountruth_metric))
        groundtruth_total += np.asarray(grountruth_metric)
    avg_metric = groundtruth_total / len(image_list)
    return avg_metric, test_save_path, score_samples


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric, test_save_path, score_samples = Inference(FLAGS)
    print(metric)
    with open(test_save_path+'../performance.txt', 'w') as f:
        for score in score_samples:
            f.writelines('{}\n'.format(score))
        f.writelines('metric is {} \n'.format(metric))