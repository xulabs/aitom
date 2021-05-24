import os
import torch
import argparse
import csv
from dataset import SubtomogramNShotDataset
from trainer import train_one_epoch, test_nets
from models import SiameseNetwork
from nets import *

parser = argparse.ArgumentParser(description='One-shot Learning and Segmentation for CECT')

# dataset
parser.add_argument('--dataset', type=str, default='snrINF', help='dataset name: snrINF / snr10000')
parser.add_argument('--data_root', type=str, default='../Data/subtomograms_snrINF/', help='data root folder')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# networks
parser.add_argument('--net', type=str, default='dusescnn', help='network of the model: scnn/dusescnn')
parser.add_argument("--N", type=int, default=2)
parser.add_argument("--K", type=int, default=1)

# training options
parser.add_argument('--n_epochs', type=int, default=40, help='number of epoch')
parser.add_argument('--batchs', type=int, default=1000, help='number of batch per epoch')
parser.add_argument('--batch_size', type=int, default=12, help='training/test batch size')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=4, help='evaluation epochs')

# optimizer
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
args = parser.parse_args()

################################################################################################
class_per_set = args.N
sample_per_class = args.K

n_epochs = args.n_epochs
batches = args.batchs
batch_size = args.batch_size

net = args.net
dataset = args.dataset
data_root = args.data_root

if not os.path.exists('outputs/{}-way-{}-shot_{}_{}'.format(class_per_set,
                                                            sample_per_class,
                                                            net,
                                                            dataset)):
    os.makedirs('outputs/{}-way-{}-shot_{}_{}'.format(class_per_set,
                                                      sample_per_class,
                                                      net,
                                                      dataset))

data = SubtomogramNShotDataset(data_root=data_root,
                               batch_size=batch_size, class_per_set=class_per_set, sample_per_class=sample_per_class)

model = SiameseNetwork(network_type=args.net).cuda()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

# training phase
continue_tarining = False
if continue_tarining:
    pre_model = ''
    print("Using Model ", pre_model, " to continue training!")
    model.load_state_dict(torch.load(pre_model))

print("====================Start Training:====================")
print("===================={}-Way-{}-Shot-{}-{} Learning====================".format(class_per_set,
                                                                                     sample_per_class,
                                                                                     net,
                                                                                     dataset))

for e in range(n_epochs):
    ac = 0.0
    dicel = 0.0

    print("====================Epoch:{}====================".format(e))
    # train_one_epoch(data, batches, model, optimizer)

    tests = 500
    for i in range(tests):
        acc, dice_loss = test_nets(data, model, batch_size)
        ac += acc
        dicel += dice_loss

    print("====================Test: test_accuracy:{} test_dice_loss:{}====================".format(ac/tests, dicel/tests))

    with open(os.path.join('outputs/{}-way-{}-shot_{}_{}'.format(class_per_set, sample_per_class, net, dataset), 'metrics_table.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow([e, ac/tests, dicel/tests])

    if e % args.eval_epochs == 0:
        torch.save(model.state_dict(), 'outputs/{}-way-{}-shot_{}_{}/model_{}.pkl'.format(class_per_set,
                                                                                          sample_per_class,
                                                                                          net,
                                                                                          dataset,
                                                                                          e))
