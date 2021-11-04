from __future__ import print_function, division
import argparse
import os
from os.path import exists, join, basename
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from util.dataloader import DataLoader # modified dataloader
from jim_net.cnn_geometric_model import CNNGeometric, TwoStageCNNGeometric, FeatureCorrelation, featureL2Norm, Jim_Net
from jim_net.loss import TransformedGridLoss, WeakInlierCount, TwoStageWeakInlierCount, ce_loss
from data.synth_dataset import SynthDataset
from data.weak_dataset import ImagePairDataset, UnsupervisedDataset
from data.pf_dataset import PFDataset, PFPascalDataset
from data.download_datasets import download_PF_pascal
from geotnf.transformation import SynthPairTnf,SynthTwoPairTnf,SynthTwoStageTwoPairTnf
from image.normalization import NormalizeImageDict, normalize_image
from jim_net_util.torch_util import save_checkpoint, str_to_bool
from jim_net_util.torch_util import BatchTensorToVars
from geotnf.transformation import GeometricTnf
from collections import OrderedDict
import numpy as np
import numpy.random
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import torch.nn.functional as F
from jim_net.cnn_geometric_model import featureL2Norm
from jim_net_util.dataloader import default_collate
from jim_net_util.eval_util import pck_metric, area_metrics, flow_metrics, compute_metric
from options.options import ArgumentParser

import sklearn
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import contingency_matrix 
from scipy.optimize import linear_sum_assignment
import imageio
import scipy.misc
from torch.utils.data import TensorDataset, DataLoader

from datetime import datetime
"""

Script to train the model using weak supervision

"""
eps = 1e-3
num_clusters = 20
compute_metric_batch = 4

iterations = 10

# train
best_test_loss = float("inf")

# Argument parsing
args,arg_groups = ArgumentParser(mode='train_weak').parse()
print(args)

use_cuda = torch.cuda.is_available()

# Seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

for _ in range(args.iterations):
    print('Jim-Net training script using weak supervision')

    # CNN model and loss
    print('Creating CNN model...')


    model = Jim_Net(use_cuda=use_cuda,
                            return_correlation=True,
                            **arg_groups['model'])

    class Clusters():
        def __init__(self):
            self.clusters = []

    cluster_class = Clusters()

    
    def print_time(message):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(message, " ", current_time)
    
    # Download validation dataset if needed
    if args.eval_dataset_path=='' and args.eval_dataset=='pf-pascal':
        args.eval_dataset_path='datasets/proposal-flow-pascal/'
    if args.eval_dataset=='pf-pascal' and not exists(args.eval_dataset_path):
        download_PF_pascal(args.eval_dataset_path)

    # load pre-trained model
    if args.model!='':
        checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
            
        for name, param in model.Alignment_Module.FeatureExtraction.state_dict().items():
            model.Alignment_Module.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
        for name, param in model.Alignment_Module.FeatureRegression.state_dict().items():
            model.Alignment_Module.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])
        for name, param in model.Alignment_Module.FeatureRegression2.state_dict().items():
            model.Alignment_Module.FeatureRegression2.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression2.' + name])
            
    if args.model_aff!='':
        checkpoint_aff = torch.load(args.model_aff, map_location=lambda storage, loc: storage)
        checkpoint_aff['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint_aff['state_dict'].items()])
        for name, param in model.Alignment_Module.FeatureExtraction.state_dict().items():
            model.Alignment_Module.FeatureExtraction.state_dict()[name].copy_(checkpoint_aff['state_dict']['FeatureExtraction.' + name])    
        for name, param in model.Alignment_Module.FeatureRegression.state_dict().items():
            model.Alignment_Module.FeatureRegression.state_dict()[name].copy_(checkpoint_aff['state_dict']['FeatureRegression.' + name])

    if args.model_tps!='':
        checkpoint_tps = torch.load(args.model_tps, map_location=lambda storage, loc: storage)
        checkpoint_tps['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint_tps['state_dict'].items()])
        for name, param in model.Alignment_Module.FeatureRegression2.state_dict().items():
            model.Alignment_Module.FeatureRegression2.state_dict()[name].copy_(checkpoint_tps['state_dict']['FeatureRegression.' + name])
            
    # set which parts of model to train      
    for name,param in model.Alignment_Module.FeatureExtraction.named_parameters():
        param.requires_grad = False
        if args.train_fe and np.sum([name.find(x)!=-1 for x in args.fe_finetune_params]):
            param.requires_grad = True        
        if args.train_fe and name.find('bn')!=-1 and np.sum([name.find(x)!=-1 for x in args.fe_finetune_params]):
            param.requires_grad = args.train_bn 
                
    for name,param in model.Alignment_Module.FeatureExtraction.named_parameters():
        print(name.ljust(30),param.requires_grad)
            
    for name,param in model.Alignment_Module.FeatureRegression.named_parameters():    
        param.requires_grad = args.train_fr 
        if args.train_fr and name.find('bn')!=-1:
            param.requires_grad = args.train_bn            

    for name,param in model.Alignment_Module.FeatureRegression2.named_parameters():    
        param.requires_grad = args.train_fr 
        if args.train_fr and name.find('bn')!=-1:
            param.requires_grad = args.train_bn

    # define loss
    print('Using weak loss...')
    if args.dilation_filter==0:
        dilation_filter = 0
    else:
        dilation_filter = generate_binary_structure(2, args.dilation_filter)
            
    inliersAffine = WeakInlierCount(geometric_model='affine',**arg_groups['weak_loss'])
    inliersTps = WeakInlierCount(geometric_model='tps',**arg_groups['weak_loss'])
    inliersComposed = TwoStageWeakInlierCount(use_cuda=use_cuda,**arg_groups['weak_loss'])


    def inlier_score_function(theta_aff,theta_aff_tps,corr_aff,corr_aff_tps,minimize_outliers=False):
        inliers_comp = inliersComposed(matches=corr_aff,
                                        theta_aff=theta_aff,
                                        theta_aff_tps=theta_aff_tps)

        inliers_aff = inliersAffine(matches=corr_aff,
                                    theta=theta_aff)
        
        inlier_score=inliers_aff+inliers_comp
        
        return inlier_score

    def loss_fun(batch):
        
        theta_aff,theta_aff_tps,corr_aff,corr_aff_tps,cluster_preds, _ =model(batch)
        
        inlier_score_pos = inlier_score_function(theta_aff,
                                                theta_aff_tps,
                                                corr_aff,
                                                corr_aff_tps)
        loss = torch.mean(-inlier_score_pos)

        loss_ce = ce_loss(cluster_preds, batch['prob'])

        return loss + args.alpha * loss_ce

    # dataset 
    train_dataset_size = args.train_dataset_size if args.train_dataset_size!=0 else None

    dataset = ImagePairDataset(csv_file=os.path.join(args.dataset_csv_path,'train_pairs.csv'),
                        training_image_path=args.dataset_image_path,
                        transform=NormalizeImageDict(['source_image','target_image']),
                        dataset_size = train_dataset_size,
                        random_crop=args.random_crop)

    dataset_eval = PFPascalDataset(csv_file=os.path.join(args.eval_dataset_path, 'val_pairs_pf_pascal.csv'),
                        dataset_path=args.eval_dataset_path,
                        transform=NormalizeImageDict(['source_image','target_image']))

    # filter training categories
    if args.categories!=0:
        keep = np.zeros((len(dataset.set),1))
        for i in range(len(dataset.set)):
            keep[i]=np.sum(dataset.set[i]==args.categories)
        keep_idx = np.nonzero(keep)[0]
        dataset.set = dataset.set[keep_idx]
        dataset.img_A_names = dataset.img_A_names[keep_idx]
        dataset.img_B_names = dataset.img_B_names[keep_idx]

    batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

    dataloader_eval = DataLoader(dataset_eval, batch_size=compute_metric_batch,
                            shuffle=False, num_workers=4)

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

    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)]) 

    warm_optimizer = torch.optim.Adam(model.Classifier.parameters(), lr=args.warmup_lr) 

    optimizer = optim.Adam([{'params':list(filter(lambda p: p.requires_grad, model.Alignment_Module.parameters()))},\
                    {'params':list(model.Dimension_Reducer1.parameters())}, \
                    {'params':list(model.Dimension_Reducer2.parameters())}, \
                    {'params':list(model.Classifier.parameters())}], lr=args.align_lr, weight_decay=args.weight_decay)

    # define epoch function

    clust_acc1 = []
    clust_acc2 = []
    #small batch seems better reached consistent
    def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,use_cuda=True,log_interval=50):
        for param_group in warm_optimizer.param_groups:
            if epoch > 1:
                param_group['lr'] = args.classifier_lr
            else:
                param_group['lr'] = 0.
        
        for i, param_group in enumerate(optimizer.param_groups):
            if epoch > 1:
                if i == 0:
                    param_group['lr'] = args.align_lr
                elif i <= 3:
                    param_group['lr'] = args.classifier_lr
                else:
                    raise NotImplementedError
            else:
                param_group['lr'] = 0.
        
        print_time("Started Feature Extraction Step: " + str(epoch))
        old_clusters = cluster_class.clusters
        #cluster step
        cluster_features = []
        images = []
        true_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                tnf_batch = batch_preprocessing_fn(batch)

                _,_,_,_, _, cluster_feature = model(tnf_batch)
                
                cluster_features.append(cluster_feature.cpu().detach().numpy())
                images.append(tnf_batch['source_image'].cpu().detach().numpy())

                #TODO make this an optional flag
                #only used for computing metrics 
                true_labels.append(tnf_batch['set'].cpu().detach().numpy())

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                tnf_batch = batch_preprocessing_fn(batch)
                _,_,_,_, _, cluster_feature = model(tnf_batch)

                cluster_features.append(cluster_feature.cpu().detach().numpy())
                images.append(tnf_batch['target_image'].cpu().detach().numpy())

                #TODO make this an optional flag
                #only used for computing metrics 
                true_labels.append(tnf_batch['set'].cpu().detach().numpy())

        cluster_features = np.concatenate(cluster_features)
        images = np.concatenate(images)
        true_labels = np.concatenate(true_labels)

        print_time("Started Clustering: " + str(epoch))

        tsne = TSNE(n_components=3, n_jobs = 40, init = 'pca', perplexity = 25)
        features_tsne = tsne.fit_transform(cluster_features)

        gmm = GaussianMixture(n_components=num_clusters, covariance_type='full')
        gmm.fit(features_tsne)
        clusters = gmm.predict(features_tsne)
        
        if len(old_clusters) != 0:
            aligned_clusters = align_cluster_index(old_clusters, clusters)
            clusters = aligned_clusters
        cluster_class.clusters = clusters

        probabilities = one_hot(clusters, num_clusters)

        print_time("Started Warm Start: " + str(epoch))
        warm_start_tensor = TensorDataset(torch.Tensor(cluster_features), torch.Tensor(probabilities))
        warm_start_loader = DataLoader(warm_start_tensor, batch_size=args.batch_size, shuffle=True)

        loss_warm_start = 0.
        for i in range(args.warmup_epochs):
            for batch_idx, (batch, label) in enumerate(warm_start_loader):
                if mode=='train':
                    optimizer.zero_grad()
                _, loss = model.classify(batch.cuda(), probabilities=label.cuda())
                loss_warm_start += loss.data.cpu().numpy()
                if mode=='train':
                    loss.backward()
                    warm_optimizer.step()
        print_time("Warm start loss: " + str(loss_warm_start/(args.warmup_epochs * len(warm_start_loader) + eps)))

        print_time("Started Alignment: " + str(epoch))
        #alignment step
        unsupervised_dataset = UnsupervisedDataset(images, cluster_features, clusters, probabilities, num_clusters)
        unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=args.batch_size, shuffle=True)
        epoch_loss = 0
        for batch_idx, batch in enumerate(unsupervised_dataloader):
            if mode=='train':
                optimizer.zero_grad()
            tnf_batch = batch_preprocessing_fn(batch)
            loss = loss_fn(tnf_batch)
            loss_np = loss.data.cpu().numpy()
            epoch_loss += loss_np
            if mode=='train':
                loss.backward()
                optimizer.step()
            else:
                loss=None
            if batch_idx % log_interval == 0:
                print(mode.capitalize()+' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                    epoch, batch_idx , len(unsupervised_dataloader),
                    100. * batch_idx / len(unsupervised_dataloader), loss_np))


        epoch_loss /= len(unsupervised_dataloader)
        print_time(mode.capitalize()+' set: Average loss: {:.4f}'.format(epoch_loss))

        #evaluate classifier
        cluster_accuracy_loader = DataLoader(torch.Tensor(images), batch_size=args.batch_size, shuffle=False)
        predicted_cluster = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(cluster_accuracy_loader):
                preds = model.classify(batch.cuda(), processed=False)
                predicted_cluster.append(preds.cpu().detach().numpy())

        pred_clusters = np.concatenate(predicted_cluster).reshape(images.shape[0], num_clusters)
        pred_clusters = np.argmax(pred_clusters, axis=1)

        print_time("V measure predictions: " + str(sklearn.metrics.v_measure_score(true_labels, pred_clusters)))
        print_time("V measure clusters: " +  str(sklearn.metrics.v_measure_score(true_labels, clusters)))
        
        try:
            aligned = align_cluster_index(true_labels - 1, pred_clusters)
            equality = np.equal(aligned, true_labels - 1)
            aligned_with_true = equality.sum()/len(true_labels)

            clust_acc1.append((epoch, aligned_with_true))
            print_time("Accuracy Predictions: " +  str(aligned_with_true))
        except:
            print_time("Failed to align predictions with labels")

        try:
            aligned = align_cluster_index(true_labels - 1, clusters)
            equality = np.equal(aligned, true_labels - 1)
            aligned_with_true = equality.sum()/len(true_labels)

            clust_acc2.append((epoch, aligned_with_true))
            print_time("Accuracy Clustering: " +  str(aligned_with_true))
        except:
            print_time("Failed to align clusters with labels")

        return epoch_loss

    # compute initial value of evaluation metric used for early stopping
    if args.eval_metric=='dist':
        metric = 'dist'
    if args.eval_metric=='pck':
        metric = 'pck'
    do_aff = args.model_aff!=""
    do_tps = args.model_tps!=""
    two_stage = args.model!='' or (do_aff and do_tps)


    if args.categories==0: 
        eval_categories = np.array(range(20))+1
    else:
        eval_categories = np.array(args.categories)
        
    eval_flag = np.zeros(len(dataset_eval))
    for i in range(len(dataset_eval)):
        eval_flag[i]=sum(eval_categories==dataset_eval.category[i])
    eval_idx = np.flatnonzero(eval_flag)

    model.eval()

    stats=compute_metric(metric,model.Alignment_Module,dataset_eval,dataloader_eval,batch_tnf,compute_metric_batch,two_stage,do_aff,do_tps,args)
    eval_value=np.mean(stats['aff_tps'][metric][eval_idx])

    print(eval_value)

    train_loss = np.zeros(args.num_epochs)
    test_loss = np.zeros(args.num_epochs)


    print('Starting training...')

    for epoch in range(1, args.num_epochs+1):
        if args.update_bn_buffers==False:
            model.eval()
        else:
            model.train()
        train_loss[epoch-1] = process_epoch('train',epoch,model,loss_fun,optimizer,dataloader,batch_tnf,log_interval=250)
        model.eval()
        stats=compute_metric(metric,model.Alignment_Module,dataset_eval,dataloader_eval,batch_tnf,compute_metric_batch,two_stage,do_aff,do_tps,args)
        eval_value=np.mean(stats['aff_tps'][metric][eval_idx])
        
        if args.eval_metric=='pck':
            test_loss[epoch-1] = -eval_value
        else:
            test_loss[epoch-1] = eval_value
            
        # remember best loss
        is_best = test_loss[epoch-1] < best_test_loss
        best_test_loss = min(test_loss[epoch-1], best_test_loss)

        print('Best test loss: ', best_test_loss)

        if is_best:
            torch.save(model.state_dict(), "./trained_models/{}.pth.tar".format(args.result_model_fn))
            
    print(clust_acc1)
    print(clust_acc2)
    print(test_loss)
