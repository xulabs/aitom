import argparse
import os
from os.path import exists, join, basename
import torch
import torch.nn.functional as F

from util.dataloader import DataLoader # modified dataloader
from jim_net.cnn_geometric_model import Jim_Net
from data.pf_dataset import PFPascalDataset
from data.download_datasets import download_PF_pascal
from image.normalization import NormalizeImageDict
from jim_net_util.torch_util import BatchTensorToVars
from geotnf.transformation import GeometricTnf
import numpy as np
import numpy.random
from jim_net_util.eval_util import compute_metric
from options.options import ArgumentParser

import sklearn
from sklearn.metrics.cluster import contingency_matrix 
from scipy.optimize import linear_sum_assignment
import imageio
from torch.utils.data import DataLoader

"""

Script to train the model using weak supervision

"""
compute_metric_batch = 4

print('WeakAlign training script using weak supervision')

# Argument parsing
args,arg_groups = ArgumentParser(mode='train_weak').parse()
print(args)

use_cuda = torch.cuda.is_available()

# Seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# CNN model and loss
print('Creating CNN model...')

model = Jim_Net(use_cuda=use_cuda,
                        return_correlation=True,
                        **arg_groups['model'])

# Download validation dataset if needed
if args.eval_dataset_path=='' and args.eval_dataset=='pf-pascal':
    args.eval_dataset_path='datasets/proposal-flow-pascal/'
if args.eval_dataset=='pf-pascal' and not exists(args.eval_dataset_path):
    download_PF_pascal(args.eval_dataset_path)

batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

dataset_test = PFPascalDataset(csv_file=os.path.join(args.eval_dataset_path, 'test_pairs_pf_pascal.csv'),
                      dataset_path=args.eval_dataset_path,
                      transform=NormalizeImageDict(['source_image','target_image']))

dataloader_test = DataLoader(dataset_test, batch_size=compute_metric_batch,
                        shuffle=False, num_workers=4)

model.load_state_dict(torch.load("./trained_models/jim_net.pth.tar"))
model.eval()

print('Final test pck...')
compute_metric('pck', model.Alignment_Module, dataset_test, dataloader_test, batch_tnf, compute_metric_batch, True, True, True, args)

true_labels = []
predicted_labels = []

saved_image_clusters = {}

def align_cluster_index(ref_cluster, map_cluster):                                                                                                                                        
    """                                                                                                                                                                            
    remap cluster index according the the ref_cluster.                                                                                                                                    
    both inputs must have same number of unique cluster index values.                                                                                                                      
    """                                                                                                                                                                                   
                                                                                                                                                                                   
    ref_values = np.unique(ref_cluster)                                                                                                                                                   
    map_values = np.unique(map_cluster)                                                                                                                                                
                                                                                                                                                                                                        
    if ref_values.shape[0]!=map_values.shape[0]:                                                                                                                                   
        print('error: both inputs must have same number of unique cluster index values.')                                                                                                      
        return()                                                                                                                                                               
    cont_mat = contingency_matrix(ref_cluster, map_cluster)                                                                                                                 
                                                                                                                                                                                                        
    row_ind, col_ind = linear_sum_assignment(len(ref_cluster) - cont_mat)                                                                                            
                                                                                                                                                                                                        
    map_cluster_out = map_cluster.copy()                                                                                                                                           
                                                                                                                                                                                                        
    for i in ref_values:                                                                                                                                                            
                                                                                                                                                                                                        
        map_cluster_out[map_cluster == col_ind[i]] = i                                                                                                                                     

    return map_cluster_out                          

#makes image clusters according to model
with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader_test):
        tnf_batch = batch_tnf(batch)

        predictions = model.classify(tnf_batch['source_image'], processed=False).cpu().detach().numpy()

        for i, prediction in enumerate(predictions):
            prediction = np.argmax(prediction)
            if prediction not in saved_image_clusters:
                saved_image_clusters[prediction] = []
            saved_image_clusters[prediction].append(tnf_batch['source_image'][i].squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy())

        predictions = model.classify(tnf_batch['target_image'], processed=False).cpu().detach().numpy()

        for i, prediction in enumerate(predictions):
            prediction = np.argmax(prediction)
            if prediction not in saved_image_clusters:
                saved_image_clusters[prediction] = []
            saved_image_clusters[prediction].append(tnf_batch['target_image'][i].squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy())

image_groups = {}

affTnf = GeometricTnf(geometric_model='affine', use_cuda=True)
def affTpsTnf(source_image, theta_aff, theta_aff_tps, use_cuda=True):
    tpstnf = GeometricTnf(geometric_model = 'tps',use_cuda=use_cuda)
    sampling_grid = tpstnf(image_batch=source_image,
                        theta_batch=theta_aff_tps,
                        return_sampling_grid=True)[1]
    X = sampling_grid[:,:,:,0].unsqueeze(3)
    Y = sampling_grid[:,:,:,1].unsqueeze(3)
    Xp = X*theta_aff[:,0].unsqueeze(1).unsqueeze(2)+Y*theta_aff[:,1].unsqueeze(1).unsqueeze(2)+theta_aff[:,2].unsqueeze(1).unsqueeze(2)
    Yp = X*theta_aff[:,3].unsqueeze(1).unsqueeze(2)+Y*theta_aff[:,4].unsqueeze(1).unsqueeze(2)+theta_aff[:,5].unsqueeze(1).unsqueeze(2)
    sg = torch.cat((Xp,Yp),3)
    warped_image_batch = F.grid_sample(source_image, sg, align_corners=True)

    return warped_image_batch 

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader_test):
        tnf_batch = batch_tnf(batch)

        classes = tnf_batch['set'].cpu().numpy()
        for i, label in enumerate(classes):
            #shift by 1 to rectify 0 vs 1 indexed
            label = label - 1
            theta_aff,theta_aff_tps,correlation_1,correlation_2,cluster_pred, dim_red = model(tnf_batch)

            source_image = tnf_batch['source_image'][i:(i + 1)]
            target_image = tnf_batch['target_image'][i:(i + 1)]

            warped_image_aff = affTnf(source_image,theta_aff[0:1].view(-1,2,3))
            warped_image_aff_tps = affTpsTnf(source_image,theta_aff[0:1], theta_aff_tps[0:1])


            source_image = source_image.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
            target_image = target_image.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
            
            warped_image_aff = warped_image_aff.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
            warped_image_aff_tps = warped_image_aff_tps.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()

            if label not in image_groups:
                image_groups[label] = []
            
            image_groups[label].append([source_image, warped_image_aff, warped_image_aff_tps, target_image])

def make_grid_clusters(images, rows=20, cols=10, margin=5):
    h, w, c = images[0][0].shape
    grid = np.zeros((margin * (rows + 1) + h * rows, margin * (cols + 1) + w * cols, c))

    for i in range(rows):
        for j in range(min(cols, len(images[i]))):
            h1 = (i + 1) * margin + i * h
            h2 = (i + 1) * margin + (i + 1) * h

            w1 = (j + 1) * margin + j * w
            w2 = (j + 1) * margin + (j + 1) * w

            grid[h1:h2, w1:w2, :] = images[i][j]

    return grid

def make_grid_transformed(images, rows=20, cols=3, margin=5, margin2=5):
    h, w, c = images[0][0][0].shape
    grid = np.zeros((margin * (rows + 1) + h * rows, margin * (4 * cols + 1) + margin2 * (cols + 1) + 4 * w * cols, c))

    for i in range(rows):
        for j in range(cols, len(images[i])):
            h1 = (i + 1) * margin + i * h
            h2 = (i + 1) * margin + (i + 1) * h

            group = images[i][j]

            if len(group) != 4:
                raise Exception

            for k in range(len(group)):
                w1 = (4 * j + k + 1) * margin + margin2 * (j + 1) + (4 * j + k) * w
                w2 = (4 * j + k + 1) * margin + margin2 * (j + 1) + (4 * j + k + 1) * w
                grid[h1:h2, w1:w2, :] = group[k]

    return grid

grid = make_grid_clusters(saved_image_clusters)
imageio.imwrite("./cluster_grid.png", grid)

grid = make_grid_transformed(image_groups)
imageio.imwrite("./alignment_grid.png", grid)        
    

true_labels = []
predicted_labels = []
for batch_idx, batch in enumerate(dataloader_test):
    tnf_batch = batch_tnf(batch)

    true_labels.append(tnf_batch['set'].cpu().detach().numpy())
    true_labels.append(tnf_batch['set'].cpu().detach().numpy())

    predictions = model.classify(tnf_batch['source_image'], processed=False).cpu().detach().numpy()
    predicted_labels.append(predictions)

    predictions = model.classify(tnf_batch['target_image'], processed=False).cpu().detach().numpy()
    predicted_labels.append(predictions)

true_labels = np.concatenate(true_labels)

predicted_labels = np.concatenate(predicted_labels)
predicted_labels = np.argmax(predicted_labels, axis=1)

aligned = align_cluster_index(true_labels - 1, predicted_labels)

equality = np.equal(aligned, true_labels - 1)
aligned_with_true = equality.sum()/len(true_labels)

print('Test Alignment Accuracy')
print(aligned_with_true)