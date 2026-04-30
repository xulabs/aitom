import torch


def iou(predictions, labels, threshold=None, average=True, device=torch.device("cpu"), classes=21,
        ignore_index=255, ignore_background=True):

    """ Calculating Intersection over Union score for semantic segmentation. """

    gt = labels.long().unsqueeze(1).to(device)

    # getting mask for valid pixels, then converting "void class" to background
    valid = gt != ignore_index
    gt[gt == ignore_index] = 0

    # converting to onehot image whith class channels
    onehot_gt_tensor = torch.LongTensor(gt.shape[0], classes, gt.shape[-2], gt.shape[-1]).zero_().to(device)
    onehot_gt_tensor.scatter_(1, gt, 1)  # write ones along "channel" dimension
    classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0

    # check if it's only background
    if ignore_background:
        only_bg = (classes_in_image[:, 1:].sum(dim=1) == 0).sum() > 0
        if only_bg:
            raise ValueError('Image only contains background. Since background is set to ' +
                             'ignored, IoU is invalid.')

    if threshold is None:
        # taking the argmax along channels
        pred = torch.argmax(predictions, dim=1).unsqueeze(1)
        pred_tensor = torch.LongTensor(pred.shape[0], classes, pred.shape[-2], pred.shape[-1]).zero_().to(device)
        pred_tensor.scatter_(1, pred, 1)
    else:
        # counting predictions above threshold
        pred_tensor = (predictions > threshold).long()

    onehot_gt_tensor *= valid.long()
    pred_tensor *= valid.long().to(device)

    intersection = (pred_tensor & onehot_gt_tensor).sum([2, 3]).float()
    union = (pred_tensor | onehot_gt_tensor).sum([2, 3]).float()

    iou = intersection / (union + 1e-12)

    start_id = 1 if ignore_background else 0

    if average:
        average_iou = iou[:, start_id:].sum(dim=1) /\
                      (classes_in_image[:, start_id:].sum(dim=1)).float()  # discard background IoU
        iou = average_iou

    return iou.cpu().numpy()


def soft_iou(pred_tensor, labels, average=True, device=torch.device("cpu"), classes=21,
             ignore_index=255, ignore_background=True):

    """ Soft IoU score for semantic segmentation, based on 10.1109/ICCV.2017.372 """

    gt = labels.long().unsqueeze(1).to(device)

    # getting mask for valid pixels, then converting "void class" to background
    valid = gt != ignore_index
    gt[gt == ignore_index] = 0
    valid = valid.float().to(device)

    # converting to onehot image with class channels
    onehot_gt_tensor = torch.LongTensor(gt.shape[0], classes, gt.shape[-2], gt.shape[-1]).zero_().to(device)
    onehot_gt_tensor.scatter_(1, gt, 1)  # write ones along "channel" dimension
    classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0
    onehot_gt_tensor = onehot_gt_tensor.float().to(device)

    # check if it's only background
    if ignore_background:
        only_bg = (classes_in_image[:, 1:].sum(dim=1) == 0).sum() > 0
        if only_bg:
            raise ValueError('Image only contains background. Since background is set to ' +
                             'ignored, IoU is invalid.')

    onehot_gt_tensor *= valid
    pred_tensor *= valid

    # intersection = torch.max(torch.tensor(0).float(), torch.min(pred_tensor, onehot_gt_tensor).sum(dim=[2, 3]))
    # union = torch.min(torch.tensor(1).float(), torch.max(pred_tensor, onehot_gt_tensor).sum(dim=[2, 3]))

    intersection = (pred_tensor * onehot_gt_tensor).sum(dim=[2, 3])
    union = (pred_tensor + onehot_gt_tensor).sum(dim=[2, 3]) - intersection

    iou = intersection / (union + 1e-12)

    start_id = 1 if ignore_background else 0

    if average:
        average_iou = iou[:, start_id:].sum(dim=1) /\
                      (classes_in_image[:, start_id:].sum(dim=1)).float()  # discard background IoU
        iou = average_iou

    return iou.cpu().numpy()


def dice_score(input, target, classes, ignore_index=-100):

    """ Functional dice score calculation. """

    target = target.long().unsqueeze(1)

    # getting mask for valid pixels, then converting "void class" to background
    valid = target != ignore_index
    target[target == ignore_index] = 0
    valid = valid.float()

    # converting to onehot image with class channels
    onehot_target = torch.LongTensor(target.shape[0], classes, target.shape[-2], target.shape[-1]).zero_().cuda()
    onehot_target.scatter_(1, target, 1)  # write ones along "channel" dimension
    # classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0
    onehot_target = onehot_target.float()

    # keeping the valid pixels only
    onehot_target = onehot_target * valid
    input = input * valid

    dice = 2 * (input * onehot_target).sum([2, 3]) / ((input**2).sum([2, 3]) + (onehot_target**2).sum([2, 3]))
    return dice.mean(dim=1)


class DiceLoss(torch.nn.Module):

    """ Dice score implemented as a nn.Module. """

    def __init__(self, classes, loss_mode='negative_log', ignore_index=255, activation=None):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.loss_mode = loss_mode
        self.activation = activation

    def forward(self, input, target):
        if self.activation is not None:
            input = self.activation(input)
        score = dice_score(input, target, self.classes, self.ignore_index)
        if self.loss_mode == 'negative_log':
            eps = 1e-12
            return (-(score+eps).log()).mean()
        elif self.loss_mode == 'one_minus':
            return (1 - score).mean()
        else:
            raise ValueError('Loss mode unknown. Please use \'negative_log\' or \'one_minus\'!')


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions = torch.softmax(torch.empty(7, 21, 224, 224).normal_(), dim=1)
    conv = torch.nn.Conv2d(21, 21, 1, 1, 0)
    labels = (torch.empty(7, 224, 224).normal_(10, 10) % 21)
    criterion = DiceLoss(21, activation=torch.nn.Softmax(dim=1))
    loss = criterion(conv(predictions), labels)

    loss.backward()
    print(loss)

    print('done.')
