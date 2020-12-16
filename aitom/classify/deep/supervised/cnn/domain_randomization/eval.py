import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dice_loss import dice_coeff


def eval_net(net, loader, device):
    """
    Evaluation without the densecrf with the dice coefficient
    """
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    # the number of batch
    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 2:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                # pred = torch.sigmoid(mask_pred)
                pred = (mask_pred[:,0] > 0.5).float()
                tot += dice_coeff(pred, true_masks[:,0].float()).item()
            pbar.update()

    net.train()
    return tot / n_val
