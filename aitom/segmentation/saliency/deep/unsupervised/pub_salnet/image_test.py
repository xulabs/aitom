import torch
from torch.utils.data import DataLoader
import torchvision
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

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
        'model_dir': "./models/state_dict/07031644/10epo_200step.ckpt",
        'logdir': "./log/",  # logdir, log on tensorboard
        'save_dir': "./save/",  # save result images as .jpg file. If None -> Not save
        'dataset': '',  # Directory of your Dataset
        'batch_size': 20,  # batchsize, default = 1
    }
    if config['logdir'] is None and config['save_dir'] is None:
        print("You should specify either config['logdir'] or config['save_dir'] to save results!")
        assert 0
    # print(os.getcwd())
    state_dict = torch.load(config['model_dir'])
    model = Unet(cfg, config['mode']).cuda()
    model.load_state_dict(state_dict)
    custom_dataset = CustomDataset(root_dir=config['dataset'])
    dataloader = DataLoader(custom_dataset, config['batch_size'], shuffle=False)
    os.makedirs(os.path.join(config['save_dir'], 'img'), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'mask'), exist_ok=True)
    if config['logdir'] is not None:
        writer = SummaryWriter(config['logdir'])
    model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        img = batch['image'].cuda()
        with torch.no_grad():
            pred, loss = model(img)
        pred = pred[5].data
        pred.requires_grad_(False)
        if config['logdir'] is not None:
            writer.add_image(config['model_dir'] + ', img', img[0], i)
            writer.add_image(config['model_dir'] + ', mask', pred[0], i)
        if config['save_dir'] is not None:
            for j in range(img.shape[0]):
                torchvision.utils.save_image(img[j], os.path.join(config['save_dir'], 'img', '{}_{}.jpg'.format(i, j)))
                torchvision.utils.save_image(pred[j],
                                             os.path.join(config['save_dir'], 'mask', '{}_{}.jpg'.format(i, j)))
    if config['logdir'] is not None:
        writer.close()
