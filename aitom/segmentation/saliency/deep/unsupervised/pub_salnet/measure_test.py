import torch
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .network import Unet
from .dataset import CustomDataset

torch.set_printoptions(profile='full')
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = {'PicaNet': "GGLLL",
           'Size': [4, 4, 4, 4, 8, 6],
           'Channel': [1024, 512, 512, 256, 128, 64],
           'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}
    config = {
        'mode': 'test',
        'model_dir': "./models/state_dict/07031644/",
        'dataset': '',  # Directory of your Dataset
        'batch_size': 20,  # batchsize, default = 1
        'logdir': "./log/",  # logdir, log on tensorboard
        'which_iter': 1,  # "Specific Iter to measure", default=-1
        'cont': 0,  # "Measure scores from this iter"
        'step': 10000,  # "Measure scores per this iter step"
    }
    models = sorted(os.listdir(config['model_dir']), key=lambda x: int(x.split('epo_')[1].split('step')[0]))
    pairdataset = CustomDataset(root_dir=config['dataset'])
    dataloader = DataLoader(pairdataset, 8, shuffle=True)
    beta_square = 0.3
    if config['logdir'] is not None:
        writer = SummaryWriter(config['logdir'])
    model = Unet(cfg, config['mode']).cuda()
    for model_name in models:
        model_iter = int(model_name.split('epo_')[1].split('step')[0])
        if model_iter % config['step'] != 0:
            continue
        if model_iter < config['cont']:
            continue
        if config['which_iter'] > 0 and config['which_iter'] != model_iter:
            continue
        state_dict = torch.load(os.path.join(config['model_dir'], model_name))
        model.load_state_dict(state_dict)
        model.eval()
        mae = 0
        preds = []
        masks = []
        precs = []
        recalls = []
        print('==============================')
        print("On iteration : " + str(model_iter))
        for i, batch in enumerate(dataloader):
            img = batch['image'].cuda()
            mask = batch['mask'].cuda()
            with torch.no_grad():
                pred, loss = model(img, mask)
            pred = pred[5].data
            mae += torch.mean(torch.abs(pred - mask))
            pred = pred.requires_grad_(False)
            preds.append(pred.cpu())
            masks.append(mask.cpu())
            prec, recall = torch.zeros(mask.shape[0], 256), torch.zeros(mask.shape[0], 256)
            pred = pred.squeeze(dim=1).cpu()
            mask = mask.squeeze(dim=1).cpu()
            thlist = torch.linspace(0, 1 - 1e-10, 256)
            for j in range(256):
                y_temp = (pred >= thlist[j]).float()
                tp = (y_temp * mask).sum(dim=-1).sum(dim=-1)
                # avoid prec becomes 0
                prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1).sum(dim=-1) + 1e-10), (tp + 1e-10) / (
                            mask.sum(dim=-1).sum(dim=-1) + 1e-10)
            # (batch, threshold)
            precs.append(prec)
            recalls.append(recall)

        prec = torch.cat(precs, dim=0).mean(dim=0)
        recall = torch.cat(recalls, dim=0).mean(dim=0)
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        print("Max F_score :", torch.max(f_score))
        print("Max_F_threshold :", thlist[torch.argmax(f_score)])
        if config['logdir'] is not None:
            writer.add_scalar("Max F_score", torch.max(f_score), global_step=model_iter)
            writer.add_scalar("Max_F_threshold", thlist[torch.argmax(f_score)], global_step=model_iter)
        pred = torch.cat(preds, 0)
        mask = torch.cat(masks, 0).round().float()
        if config['logdir'] is not None:
            writer.add_pr_curve('PR_curve', mask, pred, global_step=model_iter)
            writer.add_scalar('MAE', torch.mean(torch.abs(pred - mask)), global_step=model_iter)
        print("MAE :", torch.mean(torch.abs(pred - mask)))
        # Measure method from https://github.com/AceCoooool/DSS-pytorch solver.py
