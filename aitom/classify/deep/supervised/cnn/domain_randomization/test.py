import torch
from torch.utils.data import DataLoader
import torchvision
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from .unet3D import UNet
from .utils.dataset import CustomDataset

data_dir = ''

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = {
        'model_dir': "/home/lihuiyu/Documents/Pytorch-UNet/checkpoints/CP_epoch1.pth",
        # logdir, log on tensorboard
        'logdir': "./log/",
        # save result images as .jpg file. If None -> Not save
        'save_dir': "./output/",
        # Directory of your Dataset
        'dataset': '',
        # batchsize, default = 1
        'batch_size': 1,
    }
    if config['logdir'] is None and config['save_dir'] is None:
        print("You should specify either config['logdir'] or config['save_dir'] to save results!")
        assert 0
    net = UNet(n_channels=1, n_classes=2, bilinear=False).cuda()
    net.load_state_dict(torch.load(config['model_dir']))
    custom_dataset = CustomDataset(root_dir=config['dataset'])
    dataloader = DataLoader(custom_dataset, config['batch_size'], shuffle=False)
    os.makedirs(os.path.join(config['save_dir'], 'img'), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'mask'), exist_ok=True)
    # if config['logdir'] is not None:
    #     writer = SummaryWriter(config['logdir'])
    #     writer.close()
    net.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        img = batch['image'].cuda()
        with torch.no_grad():
            pred = net(img)
        pred = pred.data
        pred.requires_grad_(False)
        if config['save_dir'] is not None:
            for j in range(img.shape[0]):
                torchvision.utils.save_image(img[j, 0, 0], os.path.join(config['save_dir'],
                                                                        'img', '{}_{}.jpg'.format(i, j)))
                torchvision.utils.save_image(pred[j, 0, 0], os.path.join(config['save_dir'],
                                                                         'mask', '{}_{}.jpg'.format(i, j)))
