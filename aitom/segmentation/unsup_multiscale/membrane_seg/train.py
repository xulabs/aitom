from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import pdb
from medpy import metric

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d
from dataloaders.dataset import *
from networks.net_factory import net_factory
from networks.backboned_unet.unet import Unet
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='data/', help='Root directory containing the dataset')
parser.add_argument('--exp', type=str,  default='test', help='exp_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--labeled_bs', type=int, default=8, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=0, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=5, help='trained samples')
parser.add_argument('--gpu', type=str,  default='4', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
parser.add_argument('--pretrain', type=int,  default=0, help='use pretrained encoder')
args = parser.parse_args()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

image_size = (1024, 1024)
num_classes = 2

class CustomeDataset(Dataset):
    def __init__(self, base_dir, sample_list, num=None, transform=None, patch_size=512):
        self._base_dir = base_dir
        self.transform = transform
        self.patch_size = patch_size
        self.sample_list = sample_list
        self.image_dir = 'train'
        self.mask_dir = 'pseudo_GT'

        # Since each image will be split into 4 patches (2x2), expand index list
        print("Total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Map global index to image index and patch index
        case = self.sample_list[idx]
        img_path = os.path.join(self.image_dir, f'{case}.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Load mask patches
        mask_0_0 = cv2.imread(os.path.join(self.mask_dir, f'{case}_0_0.png'), cv2.IMREAD_GRAYSCALE)
        mask_0_1 = cv2.imread(os.path.join(self.mask_dir, f'{case}_0_1.png'), cv2.IMREAD_GRAYSCALE)
        mask_1_0 = cv2.imread(os.path.join(self.mask_dir, f'{case}_1_0.png'), cv2.IMREAD_GRAYSCALE)
        mask_1_1 = cv2.imread(os.path.join(self.mask_dir, f'{case}_1_1.png'), cv2.IMREAD_GRAYSCALE)

        # Merge masks: row-wise concatenation first, then stack vertically
        top = np.concatenate((mask_0_0, mask_0_1), axis=1)
        bottom = np.concatenate((mask_1_0, mask_1_1), axis=1)
        mask = np.concatenate((top, bottom), axis=0)
        
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 127).astype(np.uint8)

        sample = {'image': img, 'label': mask}

        if self.transform:
            sample = self.transform(sample)
            
        sample['image'] = torch.from_numpy(sample['image'].astype(np.float32)).unsqueeze(0)  # (1, H, W)
        sample['label'] = torch.from_numpy(sample['label'].astype(np.uint8))  # (H, W)

        return sample

def labeled_ratio_to_patients(dataset, patiens_num):
    ref_dict = None
    if "cryoET" in dataset:
        ref_dict = {"5":15, "10": 30, '100': 1484} # num samples x 4 patches
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def self_train(args, snapshot_path):
    if args.pretrain:
        model = Unet('resnet50', True, classes=num_classes).cuda()
    else:
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    
    root_path = 'train'
    sample_list = [case.replace('.jpg', '') for case in os.listdir(root_path) if int(case.split('_')[2].replace('.jpg', '')) >= 201 and int(case.split('_')[2].replace('.jpg', '')) <= 298]
    random.shuffle(sample_list)
    n_sample = len(sample_list)
    n_train = int(0.8 * n_sample)
    train_samples = sample_list[:n_train]
    val_samples = sample_list[n_train:]
    
    db_train = CustomeDataset(base_dir=args.root_path,
                            sample_list=train_samples,
                            num=None,
                            transform=transforms.Compose([RandomGenerator(image_size)]))
    db_val = CustomeDataset(base_dir=args.root_path, sample_list=val_samples)
    logging.info(f'Max labeled samples: {len(db_train)}')
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=args.labeled_bs, num_workers=4, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=args.labeled_bs, shuffle=False, num_workers=4)
    print('Number of train samples: ', len(db_train))
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
    
    DICE = losses.mask_DiceLoss(nclass=2)
    
    model.train()
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_performance = 0.0
    max_iterations = args.max_iterations
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            if args.pretrain:
                volume_batch = volume_batch.repeat(1, 3, 1, 1)

            # Step 2: forward pass
            outputs = model(volume_batch)

            loss_ce = F.cross_entropy(outputs, label_batch.long())
            loss_dice = DICE(outputs, label_batch.long())
            loss = (loss_ce + loss_dice) / 2
            # loss = loss_ce

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            logging.info('iteration %d : loss: %03f, loss_ce: %03f'%(iter_num, loss, loss_ce))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                performance = val_2d.test_2d_dataset(args, valloader, model, len(db_val), num_classes)

                if performance > best_performance:
                    best_performance = performance
                    # save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    # save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()
                
                save_path = os.path.join(snapshot_path,'{}_latest_model.pth'.format(args.model))
                save_net_opt(model, optimizer, save_path)
                
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break


if __name__ == "__main__":
    ## make logger file
    self_snapshot_path = "../model/supervised/{}_{}_labeled/".format(args.exp, args.labelnum)
    print("Starting BCP training.")
    for snapshot_path in [self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/train.py', self_snapshot_path)
    
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, self_snapshot_path)