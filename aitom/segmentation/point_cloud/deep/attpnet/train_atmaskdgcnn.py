from __future__ import print_function
import sys

import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from .modelnet_dataset import ModelNetDataLoader, load_data
import torch.nn.functional as F
from tqdm import tqdm
from .atmask_dgcnn import ATMASKDGCNN as WholeModel
from .atmask_dgcnn import cal_loss

sys.path.insert(0, '../pointnet/')

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')

opt = parser.parse_args()
opt.LR = 0.001
opt.workers = 4
opt.nepoch = 251
opt.batchSize = 24
opt.num_points = 1024
opt.outf = '../saved_model/atmaskdgcnn_saved_models'
opt.dataset = '../../modelnet40_ply_hdf5_2048/'
# opt.dataset_type = 'myMolecule'
opt.dataset_type = 'modelnet40'
opt.model = '../saved_model/atmaskdgcnn_saved_models/cls_model_5.pth'
# opt.model = ''
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(root=opt.dataset, classification=True, npoints=opt.num_points,
                              data_augmentation=False)

    test_dataset = ShapeNetDataset(root=opt.dataset, classification=True, split='test', npoints=opt.num_points,
                                   data_augmentation=False)

elif opt.dataset_type == 'modelnet40':
    ROTATION = None
    train_data, train_label, test_data, test_label = load_data(opt.dataset, classification=True)
    dataset = ModelNetDataLoader(train_data, train_label, rotation=ROTATION, data_augmentation=True)
    test_dataset = ModelNetDataLoader(test_data, test_label, rotation=ROTATION, data_augmentation=False)

else:
    dataset = MyMoleculeDataset(root_path=opt.dataset, npoints=opt.num_points, split='train')

    test_dataset = MyMoleculeDataset(root_path=opt.dataset, split='test', npoints=opt.num_points,
                                     data_augmentation=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
                                         num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True,
                                             num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = 40
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = WholeModel()
classifier.to(device)

if opt.model != '':
    # from collections import OrderedDict
    # state_dict = torch.load(opt.model, map_location='cpu')
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    new_state_dict = torch.load(opt.model)
    classifier.load_state_dict(new_state_dict)

# optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
optimizer = optim.SGD(classifier.parameters(), lr=opt.LR * 100, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch, eta_min=opt.LR)
num_batch = len(dataset) / opt.batchSize
loss_fn = cal_loss

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0].long()
        tmp_bs = target.shape[0]
        points = points.transpose(2, 1).to(device)
        target = target.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred = classifier(points, None)
        loss = cal_loss(pred, target)

        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, loss.item(), correct.item() / float(tmp_bs)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0].long()
            tmp_bs = target.shape[0]
            points = points.transpose(2, 1).to(device)
            target = target.to(device)
            classifier = classifier.eval()
            pred = classifier(points, None)
            loss = cal_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(tmp_bs)))
    if epoch % 5 == 0:
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
each_prec = [0.0] * num_classes
each_deno = [0] * num_classes
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0].long()
    points = points.transpose(2, 1).to(device)
    target = target.to(device)
    classifier = classifier.eval()
    pred = classifier(points, None)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    for ix in range(len(pred_choice)):
        each_deno[pred_choice[ix]] += 1
        if pred_choice[ix] == target.data[ix]:
            each_prec[pred_choice[ix]] += 1
    total_correct += correct.item()
    total_testset += points.size()[0]

for ix in range(len(each_prec)):
    each_prec[ix] = each_prec[ix] / (each_deno[ix] + 0.000001)

print("final accuracy {}".format(total_correct / float(total_testset)))
print('Class Acc:')
for i in range(num_classes):
    print("%.4f " % (each_prec[i]), end=' ')
print("\nmA: %.4f" % (sum(each_prec) / len(each_prec)))
