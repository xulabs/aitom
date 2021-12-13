
# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR10

# Custom
import models.resnet as resnet
from config import *
from data.sampler import SubsetSequentialSampler


##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

cifar10_train = CIFAR10('', train=True, download=False, transform=train_transform)    # specify data path
cifar10_unlabeled = CIFAR10('', train=True, download=False, transform=test_transform)
cifar10_test = CIFAR10('', train=False, download=False, transform=test_transform)


##
# Train Utils
iters = 0

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch):
    models['backbone'].train()
    global iters

    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()

        scores = models['backbone'](inputs)[0]
        target_loss = criterion(scores, labels)

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_backbone_loss.backward()

        optimizers['backbone'].step()

#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores = models['backbone'](inputs)[0]
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, cycle):
    print('>> Train a Model...')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):

        train_epoch(models, criterion, optimizers, dataloaders, epoch)
        schedulers['backbone'].step()

        # Save a checkpoint
        if epoch % 20 == 0 or epoch == 199:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict()
                },
                '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
            print('Cycle:', cycle, 'Epoch:', epoch, '---', 'Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc), flush=True)
    print('>> Finished.')

#
def compute_gradnorm(models, loss):
    grad_norm = torch.tensor([]).cuda()
    gradnorm = 0.0

    models['backbone'].zero_grad()
    loss.backward(retain_graph=True)
    for param in models['backbone'].parameters():
        if param.grad is not None:
            gradnorm = torch.norm(param.grad)
            gradnorm = gradnorm.unsqueeze(0)
            grad_norm = torch.cat((grad_norm, gradnorm), 0)

    return grad_norm

#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    uncertainty = torch.tensor([]).cuda()

    criterion = nn.CrossEntropyLoss()

    for j in range(1):
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            scores = models['backbone'](inputs)[0]
            posterior = F.softmax(scores, dim=1)

            loss = 0.0

            if SCHEME == 0:   # expected-gradnorm
                posterior = posterior.squeeze()

                for i in range(NUM_CLASS):
                    label = torch.full([1], i)
                    label = label.cuda()
                    loss += posterior[i] * criterion(scores, label)

            if SCHEME == 1:  # entropy-gradnorm
                loss = Categorical(probs=posterior).entropy()

            pred_gradnorm = compute_gradnorm(models, loss)
            pred_gradnorm = torch.sum(pred_gradnorm)
            pred_gradnorm = pred_gradnorm.unsqueeze(0)

            uncertainty = torch.cat((uncertainty, pred_gradnorm), 0)

    return uncertainty.cpu()


##
# Main
if __name__ == '__main__':

    for trial in range(TRIALS):
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        START = 2 * ADDENDUM
        labeled_set = indices[:START]
        unlabeled_set = indices[START:]

        train_loader = DataLoader(cifar10_train, batch_size=BATCH,     
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(cifar10_test, batch_size=BATCH)

        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        backbone_net = resnet.ResNet18(NUM_CLASS).cuda()

        models = {'backbone': backbone_net}
        torch.backends.cudnn.benchmark = True

        # Active learning cycles
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, cycle)
            acc = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc), flush=True)


            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=1,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)


            # Estimate uncertainty for unlabeled samples
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)


            # Update the labeled pool and unlabeled pool, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())       
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled pool
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH,    
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        # Save a checkpoint
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict()
                },
                './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))

        print('---------------------------Current Trial is done-----------------------------', flush=True)

