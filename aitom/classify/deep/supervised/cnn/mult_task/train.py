import torch
from torch.utils.data import DataLoader
import torchvision
import datetime
import time
import os
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .network import SSN3DED
from .dataset import CustomDataset

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = {
        'mode': 'train', # ['train','test']
        'lr': 0.001, # learning_rate, default = 0.001
        'lr_decay': 0.1, # Learning rate decrease by lr_decay time per decay_step, default = 0.1
        'decay_step': 7000, # Learning rate decrease by lr_decay time per decay_step, default = 7000
        'batch_size': 20, # batchsize, default = 1
        'epoch': 20, # epochs, default = 20
        'dataset': '', # Directory of your Dataset
        'load': None, # Directory of pre-trained model
    }
    torch.cuda.manual_seed_all(1234)
    torch.manual_seed(1234)
    start_iter = 0
    now = datetime.datetime.now()
    start_epo = 0

    duts_dataset = CustomDataset(config['dataset'])
    model = SSN3DED(config['mode']).cuda()
    # vgg = torchvision.models.vgg16(pretrained=True)
    # model.encoder.seq.load_state_dict(vgg.features.state_dict())
    # del vgg

    if config['load'] is not None:
        state_dict = torch.load(config['load'], map_location='cuda')

        start_iter = 1 # int(config['load'].split('epo_')[1].strip('step.ckpt')) + 1
        start_epo = 0 # int(config['load'].split('/')[3].split('epo')[0])
        now = time.strftime('%Y%m%d-%H%M%S', time.localtime())

        print("Loading Model from {}".format(config['load']))
        print("Start_iter : {}".format(start_iter))
        model.load_state_dict(state_dict)
        for cell in model.decoder:
            if cell.mode == 'G':
                cell.picanet.renet.vertical.flatten_parameters()
                cell.picanet.renet.horizontal.flatten_parameters()
        print('Loading_Complete')

    # Optimizer Setup
    learning_rate = config['lr'] * (config['lr_decay'] ** (start_iter // config['decay_step']))
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate * 10,
                                momentum=0.9,
                                weight_decay=0.0005)
    # Dataloader Setup
    dataloader = DataLoader(duts_dataset, config['batch_size'], shuffle=True, num_workers=0)
    # Logger Setup
    os.makedirs(os.path.join('log', now.strftime('%m%d%H%M')), exist_ok=True)
    weight_save_dir = os.path.join('models', 'state_dict', now.strftime('%m%d%H%M'))
    os.makedirs(os.path.join(weight_save_dir), exist_ok=True)
    writer = SummaryWriter(os.path.join('log', now.strftime('%m%d%H%M')))
    iterate = start_iter
    for epo in range(start_epo, config['epoch']):
        print("\nEpoch : {}".format(epo))
        for i, batch in enumerate(tqdm(dataloader)):
            if i > 10:
                break
            optimizer.zero_grad()
            img = batch['image'].cuda()
            mask = batch['mask'].cuda()
            pred, loss = model(img, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()
            writer.add_scalar('loss', float(loss), global_step=iterate)

            if iterate % 200 == 0:
                if i != 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(weight_save_dir, '{}epo_{}step.ckpt'.format(epo, iterate)))
            if iterate % 1000 == 0 and i != 0:
                for file in weight_save_dir:
                    if '00' in file and '000' not in file:
                        os.remove(os.path.join(weight_save_dir, file))
            if i + epo * len(dataloader) % config['decay_step'] == 0 and i != 0:
                learning_rate *= config['lr_decay']
                opt_en = torch.optim.SGD(model.encoder.parameters(),
                                         lr=learning_rate,
                                         momentum=0.9,
                                         weight_decay=0.0005)
                opt_dec = torch.optim.SGD(model.decoder.parameters(),
                                          lr=learning_rate * 10,
                                          momentum=0.9,
                                          weight_decay=0.0005)
            iterate += config['batch_size']
            del loss
        start_iter = 0
