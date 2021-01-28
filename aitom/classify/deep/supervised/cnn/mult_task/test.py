import torch
from torch.utils.data import DataLoader
import torchvision
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from .network import SSN3DED
from .dataset import CustomDataset

torch.set_printoptions(profile='full')
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = {
        'mode': 'test', # ['train','test']
        'model_dir': "./models/state_dict/07052156/10epo_200step.ckpt",
        'logdir': "./log/", # logdir, log on tensorboard
        'save_dir': "./save/", # save result images as .jpg file. If None -> Not save
        'dataset': '', # Directory of your Dataset
        'batch_size': 20, # batchsize, default = 1
    }
    if config['logdir'] is None and config['save_dir'] is None:
        print("You should specify either config['logdir'] or config['save_dir'] to save results!")
        assert 0
    # print(os.getcwd())
    state_dict = torch.load(config['model_dir'])
    model = SSN3DED(config['mode']).cuda()
    model.load_state_dict(state_dict)
    custom_dataset = CustomDataset(root_dir=config['dataset'])
    dataloader = DataLoader(custom_dataset, config['batch_size'], shuffle=False)
    os.makedirs(os.path.join(config['save_dir'], 'img'), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'mask'), exist_ok=True)
    if config['logdir'] is not None:
        writer = SummaryWriter(config['logdir'])
        writer.close()
    model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        img = batch['image'].cuda()
        with torch.no_grad():
            pred = model(img)
        pred = pred[0].data
        pred.requires_grad_(False)
        if config['save_dir'] is not None:
            for j in range(img.shape[0]):
                torchvision.utils.save_image(
                    img[j, 0, 0], os.path.join(config['save_dir'], 'img', '{}_{}.jpg'.format(i, j)))
                torchvision.utils.save_image(
                    pred[j, 0, 0], os.path.join(config['save_dir'], 'mask',
                                                '{}_{}.jpg'.format(i, j)))
