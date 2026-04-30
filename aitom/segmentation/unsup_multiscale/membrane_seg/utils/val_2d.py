import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import pdb



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0


def test_single_volume(image, label, model, classes):
    image = image.squeeze(0)
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        input = slice.unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            pred = out.cpu().detach().numpy()
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_single_volume_cross(image, label, model_l, model_r, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model_r.eval()
        model_l.eval()
        with torch.no_grad():
            output_l = model_l(input)
            output_r = model_r(input)
            output = (output_l + output_r) / 2
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_2d_dataset(args, valloader, model, num_samples, num_classes):
    metric_list = []

    def calculate_metric_percase(pred, gt):
        pred = (pred > 0).astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)

        if pred.sum() > 0 and gt.sum() > 0:
            return metric.binary.dc(pred, gt)
        elif pred.sum() == 0 and gt.sum() == 0:
            return 1.0  # No prediction and no GT → perfect match
        else:
            return 0.0  # One empty, one not → worst case

    model.eval()
    with torch.no_grad():
        for sampled_batch in valloader:
            images = sampled_batch["image"]  # (B, 1, H, W)
            labels = sampled_batch["label"]  # (B, H, W) or (B, 1, H, W)
            
            if args.pretrain:
                images = images.repeat(1, 3, 1, 1)

            # Move batch to GPU
            images = images.float().cuda(non_blocking=True)  # (B, 1, H, W)

            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # (B, H, W)

            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            for pred, gt in zip(predictions, labels):
                if gt.ndim == 3:
                    gt = gt.squeeze()
                for class_idx in range(1, num_classes):
                    metric_list.append(calculate_metric_percase(pred == class_idx, gt == class_idx))

    if len(metric_list) == 0:
        return 0.0
    return np.mean(metric_list)
    
