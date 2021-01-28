import argparse
import torch
from .models import main_models
# from models.main_models import CORAL
import numpy as np
import math
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .CECT_dataloader import CECT_dataset
from . import CECT_dataloader
import os
from .CORAL import CORAL
import time

# from CORAL import CORAL_pair
parser = argparse.ArgumentParser()
parser.add_argument('--n_epoches_1', type=int, default=20)
parser.add_argument('--n_epoches_1.5', type=int, default=20)
parser.add_argument('--n_epoches_2', type=int, default=100)

parser.add_argument('--n_epoches_3', type=int, default=30)
parser.add_argument('--n_target_samples', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--CORAL_batch_size', type=int, default=128)
parser.add_argument('--load_model', type=int, default=0)  # load_model=[0,1,2,3], means read which stage of the model
parser.add_argument('--classes_num', type=int, default=7)
parser.add_argument('--src_data', type=str, default='../../simulator/N_datasetA_0_5')
parser.add_argument('--tar_data', type=str, default='../..//simulator/experimental/Noble')
# model parameter
parser.add_argument('--encoder_hid_dim', type=int, default=16)
parser.add_argument('--encoder_z_dim', type=int, default=128)
parser.add_argument('--classifier_input_dim', type=int, default=128)
# 上面的参数为： shape = 28时：16 128 128 or
# shape=40时：32 32 256
opt = vars(parser.parse_args())

use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:1') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
cectA_dataset = CECT_dataset(path=opt['tar_data'])
cectA_dataloader = DataLoader(dataset=cectA_dataset, batch_size=opt['batch_size'], shuffle=True)
cectB_dataset = CECT_dataset(path=opt['src_data'])
cectB_dataloader = DataLoader(dataset=cectB_dataset, batch_size=opt['batch_size'], shuffle=True)

stage = opt['load_model']
classifier = main_models.Classifier(opt)
encoder1_S = main_models.Encoder1(opt)
encoder1_T = main_models.Encoder1(opt)
encoder2 = main_models.Encoder2(opt)
discriminator = main_models.DCD(opt)
conv_discriminator = main_models.CONV_DCD(opt)

classifier.to(device)
encoder1_S.to(device)
encoder1_T.to(device)
encoder2.to(device)
discriminator.to(device)
conv_discriminator.to(device)

if not os.path.exists('results'):
    os.mkdir('results')
if stage == 1:
    print('We will skip stage 1, stage 2 will be processed.')
    encoder1_S.load_state_dict(torch.load('results/encoder1S_stage1.pt'))
    encoder1_T.load_state_dict(torch.load('results/encoder1T_stage1.pt'))
    encoder2.load_state_dict(torch.load('results/encoder2_stage1.pt'))
    classifier.load_state_dict(torch.load('results/classifier_stage1.pt'))
elif stage == 2:
    print('We will skip stage 2, stage 3 will be processed.')
    encoder1_S.load_state_dict(torch.load('results/encoder1S_stage1.pt'))
    encoder1_T.load_state_dict(torch.load('results/encoder1T_stage1.pt'))
    encoder2.load_state_dict(torch.load('results/encoder2_stage1.pt'))
    classifier.load_state_dict(torch.load('results/classifier_stage1.pt'))
    discriminator.load_state_dict(torch.load('results/discriminator_stage2.pt'))
    conv_discriminator.load_state_dict(torch.load('results/conv_discriminator_stage2.pt'))
elif stage == 3:
    print('We will skip training, evaluation will be processed.')
    encoder1_S.load_state_dict(torch.load('results/encoder1S_stage3.pt'))
    encoder1_T.load_state_dict(torch.load('results/encoder1T_stage3.pt'))
    encoder2.load_state_dict(torch.load('results/encoder2_stage3.pt'))
    classifier.load_state_dict(torch.load('results/classifier_stage3.pt'))
    discriminator.load_state_dict(torch.load('results/discriminator_stage3.pt'))
    conv_discriminator.load_state_dict(torch.load('results/conv_discriminator_stage3.pt'))
else:
    print('We will train models without loading any pretrained models.')

begin = time.time()
# #--------------pretrain g and h for step 1---------------------------------
optimizer = torch.optim.Adam(
    list(encoder2.parameters()) + list(encoder1_S.parameters()) + list(classifier.parameters()), lr=0.001)

loss_fn = torch.nn.CrossEntropyLoss()
if stage < 1:
    for epoch in range(opt['n_epoches_1']):
        for data, labels in cectB_dataloader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            _, encoder_vectors = encoder2(encoder1_S((data)))
            y_pred = classifier(encoder_vectors)

            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

        acc = 0
        for data, labels in cectA_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            _, encoder_vectors = encoder2(encoder1_S((data)))
            y_test_pred = classifier(encoder_vectors)
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().sum().item()
        accuracy = round(acc / float(len(cectA_dataset)), 3)

        print("step1----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, opt['n_epoches_1'], accuracy))

    # -----------------train CORAL for step 1.5--------------------------------
    encoder1_T.load_state_dict(encoder1_S.state_dict())
    optimizer = torch.optim.Adam(
        list(encoder2.parameters()) + list(encoder1_S.parameters()) + list(encoder1_T.parameters()) + list(
            classifier.parameters()), lr=0.0002)

    for epoch in range(opt['n_epoches_1.5']):
        X_s_batch, Y_s_batch = CECT_dataloader.build_datapair(opt['CORAL_batch_size'], opt['src_data'])
        X_t_batch, Y_t_batch = CECT_dataloader.build_datapair(opt['CORAL_batch_size'], opt['tar_data'])
        # n=len(X_s_batch)
        n = min(len(X_s_batch), len(X_t_batch))
        print('training CORAL, ', round(epoch * 100 / opt['n_epoches_1.5'], 3), '%')
        for i in range(n):
            X_s = X_s_batch[i].to(device)
            Y_s = Y_s_batch[i].to(device)
            X_t = X_t_batch[i].to(device)
            Y_t = Y_t_batch[i].to(device)

            optimizer.zero_grad()

            _, encoder_X_s = encoder2(encoder1_S((X_s)))
            _, encoder_X_t = encoder2(encoder1_T((X_t)))
            y_pred = classifier(encoder_X_s)
            CORAL_loss = CORAL(encoder_X_s, encoder_X_t, device)
            loss = loss_fn(y_pred, Y_s)
            all_loss = loss + CORAL_loss * 500
            all_loss.backward()
            optimizer.step()
    torch.save(encoder1_S.state_dict(), 'results/encoder1S_stage1.pt')
    torch.save(encoder1_T.state_dict(), 'results/encoder1T_stage1.pt')
    torch.save(encoder2.state_dict(), 'results/encoder2_stage1.pt')
    torch.save(classifier.state_dict(), 'results/classifier_stage1.pt')
    test_dataset = CECT_dataset(path=opt['tar_data'])
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt['batch_size'], shuffle=True)

    acc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = classifier(encoder2(encoder1_T((data)))[1])
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

    accuracy = round(acc / float(len(test_dataloader)), 3)

    print("accuracy: %.3f " % accuracy)

# -----------------train DCD for step 2--------------------------------
X_s, Y_s = CECT_dataloader.sample_data(opt)
X_t, Y_t = CECT_dataloader.create_target_samples(opt, opt['n_target_samples'], classes_num=opt['classes_num'])

# X_s_batch, _ = CECT_dataloader.build_datapair(1,opt['src_data'])
# X_t_batch, _ = CECT_dataloader.build_datapair(1,opt['tar_data'])
# Ds = []
# Dt = []
# for i in range(len(X_s_batch)):
#     Ds.append(np.array(encoder(X_s_batch[i].to(device)).squeeze(0).detach().cpu()))
# for i in range(len(X_t_batch)):
#     Dt.append(np.array(encoder(X_t_batch[i].to(device)).squeeze(0).detach().cpu()))
# Ds = np.array(Ds)
# Dt = np.array(Dt)

# Cs, Cs_inv, Ct, Ct_inv = CORAL_pair(Ds,Dt)
# Cs = Cs.to(device)
# Cs_inv = Cs_inv.to(device)
# Ct = Ct.to(device)
# Ct_inv = Ct_inv.to(device)

optimizer_D = torch.optim.Adam(list(conv_discriminator.parameters()) + list(discriminator.parameters()), lr=0.0002)
# optimizer_F=torch.optim.Adam(conv_discriminator.parameters(),lr=0.001)
if stage < 2:
    for epoch in range(opt['n_epoches_2']):
        # data
        groups, _ = CECT_dataloader.sample_groups(X_s, Y_s, X_t, Y_t, seed=epoch)

        n_iters = 4 * len(groups[1])
        index_list = torch.randperm(n_iters)
        mini_batch_size = 40  # use mini_batch train can be more stable

        loss_mean = []

        X1 = []
        X2 = []
        ground_truths = []
        for index in range(n_iters):

            ground_truth = math.floor(index_list[index] / len(groups[1]))

            x1, x2 = groups[ground_truth][index_list[index] - len(groups[1]) * ground_truth]
            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            # select data for a mini-batch to train
            if (index + 1) % mini_batch_size == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_D.zero_grad()
                # optimizer_F.zero_grad()
                X1_feature = encoder1_S(X1)
                X2_feature = encoder1_T(X2)
                X_cat_feature = torch.cat([X1_feature, X2_feature], 1)
                _, encoder_X_s = encoder2(X1_feature)
                _, encoder_X_t = encoder2(X2_feature)
                X_cat = torch.cat([encoder_X_s, encoder_X_t], 1)
                encoder_feature = conv_discriminator(X_cat_feature.detach())
                y_pred = discriminator(X_cat.detach(), encoder_feature)

                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                # loss2=loss_fn(y_pred_2,ground_truths)
                optimizer_D.step()
                # optimizer_F.step()
                loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []

        print("step2----Epoch %d/%d loss:%.3f" % (epoch + 1, opt['n_epoches_2'], np.mean(loss_mean)))

    torch.save(discriminator.state_dict(), 'results/discriminator_stage2.pt')
    torch.save(conv_discriminator.state_dict(), 'results/conv_discriminator_stage2.pt')
# ----------------------------------------------------------------------

# -------------------training for step 3-------------------
# optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.001)
# optimizer_g_h=torch.optim.Adam(list(encoder2.parameters())+list(encoder1_S.parameters())+list(encoder1_T.parameters())+list(classifier.parameters()),lr=0.001)
optimizer_g_h = torch.optim.Adam(
    list(encoder2.parameters()) + list(encoder1_S.parameters()) + list(encoder1_T.parameters()) + list(
        classifier.parameters()), lr=0.001)

test_dataset = CECT_dataset(path=opt['tar_data'])
test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt['batch_size'], shuffle=True)

max_acc = 0

if stage < 3:

    for epoch in range(opt['n_epoches_3']):
        # X_s_batch, Y_s_batch = CECT_dataloader.build_datapair(opt,opt['src_data'])
        # X_t_batch, Y_t_batch = CECT_dataloader.build_datapair(opt,opt['tar_data'])
        # n=len(X_s_batch)
        # for i in range(n):
        #     X_s_c = X_s_batch[i].to(device)
        #     Y_s_c = Y_s_batch[i].to(device)
        #     X_t_c = X_t_batch[i].to(device)
        #     Y_t_c = Y_t_batch[i].to(device)

        #     optimizer.zero_grad()

        #     encoder_X_s = encoder(X_s_c)
        #     encoder_X_t = encoder(X_t_c)
        #     y_pred=classifier(encoder_X_s)
        #     CORAL_loss = CORAL(encoder_X_s,encoder_X_t,device)
        #     loss=loss_fn(y_pred,Y_s_c)
        #     all_loss = loss + CORAL_loss * 500
        #     all_loss.backward()
        #     optimizer.step()

        # ---training g and h , DCD is frozen
        groups, groups_y = CECT_dataloader.sample_groups(X_s, Y_s, X_t, Y_t, seed=opt['n_epoches_2'] + epoch)
        G1, G2, G3, G4 = groups
        Y1, Y2, Y3, Y4 = groups_y
        groups_2 = [G2, G4]
        groups_y_2 = [Y2, Y4]

        n_iters = 2 * len(G2)
        index_list = torch.randperm(n_iters)

        n_iters_dcd = 4 * len(G2)
        index_list_dcd = torch.randperm(n_iters_dcd)

        # data only contains G2 and G4 ,so decrease mini_batch
        mini_batch_size_g_h = 20
        # data contains G1,G2,G3,G4 so use 40 as mini_batch
        mini_batch_size_dcd = 40
        X1 = []
        X2 = []
        ground_truths_y1 = []
        ground_truths_y2 = []
        dcd_labels = []
        for index in range(n_iters):
            ground_truth = math.floor(index_list[index] / len(G2))
            x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            # y1=torch.LongTensor([y1.item()])
            # y2=torch.LongTensor([y2.item()])
            dcd_label = 0 if ground_truth == 0 else 2
            X1.append(x1)
            X2.append(x2)
            ground_truths_y1.append(y1)
            ground_truths_y2.append(y2)
            dcd_labels.append(dcd_label)

            if (index + 1) % mini_batch_size_g_h == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths_y1 = torch.LongTensor(ground_truths_y1)
                ground_truths_y2 = torch.LongTensor(ground_truths_y2)
                dcd_labels = torch.LongTensor(dcd_labels)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths_y1 = ground_truths_y1.to(device)
                ground_truths_y2 = ground_truths_y2.to(device)
                dcd_labels = dcd_labels.to(device)

                optimizer_g_h.zero_grad()

                # prototypes = [[] for _ in range(opt['classes_num'])]
                # # attention = main_models.MultiHeadAttention(1, opt['classifier_input_dim'], opt['classifier_input_dim'], opt['classifier_input_dim'], dropout=0)
                # # attention = attention.to(device)
                # for i in range(len(X_t)):
                #     prototypes[int(Y_t[i])].append(np.array(encoder(X_t[i].unsqueeze(0).to(device)).detach().cpu()))
                # prototypes = torch.Tensor(prototypes).mean(1)
                # prototypes.to(device)

                X1_feature = encoder1_S(X1)
                X2_feature = encoder1_T(X2)
                _, encoder_X1 = encoder2(X1_feature)
                _, encoder_X2 = encoder2(X2_feature)
                # y_prototypes = main_models.prototype_loss(encoder_X1.mm(Cs_inv).mm(Ct),prototypes.to(device))
                # loss_proto=loss_fn(y_prototypes,ground_truths_y1)

                # CORAL_Loss=CORAL(encoder_X1,encoder_X2,device)
                X_cat = torch.cat([encoder_X1, encoder_X2], 1)
                y_pred_X1 = classifier(encoder_X1)
                y_pred_X2 = classifier(encoder_X2)

                X_cat_feature = torch.cat([X1_feature, X2_feature], 1)
                encoder_feature = conv_discriminator(X_cat_feature)
                y_pred_dcd = discriminator(X_cat, encoder_feature)

                loss_X1 = loss_fn(y_pred_X1, ground_truths_y1)
                loss_X2 = loss_fn(y_pred_X2, ground_truths_y2)
                loss_dcd = loss_fn(y_pred_dcd, dcd_labels)
                # loss_dcd_conv=loss_fn(y_pred_dcd_donv,dcd_labels)
                loss_sum = loss_X1 + loss_X2 + 0.2 * loss_dcd

                loss_sum.backward()
                optimizer_g_h.step()

                X1 = []
                X2 = []
                ground_truths_y1 = []
                ground_truths_y2 = []
                dcd_labels = []

        # #----training dcd ,g and h frozen
        # X1 = []
        # X2 = []
        # ground_truths = []
        # for index in range(n_iters_dcd):

        #     ground_truth=math.floor(index_list_dcd[index]/len(groups[1]))

        #     x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
        #     X1.append(x1)
        #     X2.append(x2)
        #     ground_truths.append(ground_truth)

        #     if (index + 1) % mini_batch_size_dcd == 0:
        #         X1 = torch.stack(X1)
        #         X2 = torch.stack(X2)
        #         ground_truths = torch.LongTensor(ground_truths)
        #         X1 = X1.to(device)
        #         X2 = X2.to(device)
        #         ground_truths = ground_truths.to(device)

        #         optimizer_d.zero_grad()
        #         _, encoder_X_s = encoder2(encoder1_S((X1)))
        #         _, encoder_X_t = encoder2(encoder1_T((X2)))
        #         X_cat=torch.cat([encoder_X_s,encoder_X_t],1)
        #         y_pred = discriminator(X_cat.detach())
        #         loss = loss_fn(y_pred, ground_truths)
        #         loss.backward()
        #         optimizer_d.step()
        #         # loss_mean.append(loss.item())
        #         X1 = []
        #         X2 = []
        #         ground_truths = []

        # testing
        acc = 0

        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)

            _, encoder_data = encoder2(encoder1_T((data)))

            y_test_pred = classifier(encoder_data)
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

        accuracy = round(acc / float(len(test_dataloader)), 3)
        print("step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, opt['n_epoches_3'], accuracy))
        if accuracy > max_acc:
            max_acc = accuracy
            print('best model, saving…')
            torch.save(encoder1_S.state_dict(), 'results/encoder1S_stage3.pt')
            torch.save(encoder1_T.state_dict(), 'results/encoder1T_stage3.pt')
            torch.save(encoder2.state_dict(), 'results/encoder2_stage3.pt')
            torch.save(classifier.state_dict(), 'results/classifier_stage3.pt')
            torch.save(discriminator.state_dict(), 'results/discriminator_stage3.pt')
            torch.save(conv_discriminator.state_dict(), 'results/conv_discriminator_stage3.pt')

print("max_acc is %f" % max_acc)

result_log = opt['src_data'].split('/')[-1] + ' ' + opt['tar_data'].split('/')[-1] + ' sample=%d :%f' % (
    opt['n_target_samples'], max_acc)
os.system("echo %s >> log" % result_log)

# #-------------------training for step 4-------------------
# prototypes = [[] for _ in range(opt['classes_num'])]
# # attention = main_models.MultiHeadAttention(1, opt['classifier_input_dim'], opt['classifier_input_dim'], opt['classifier_input_dim'], dropout=0)
# # attention = attention.to(device)
# for i in range(len(X_t)):
#     prototypes[int(Y_t[i])].append(np.array(encoder(X_t[i].unsqueeze(0).to(device)).detach().cpu()))
# prototypes = torch.Tensor(prototypes).mean(1)
# prototypes.to(device)

# optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.0002)


# for epoch in range(opt['n_epoches_1']):
#         for data,labels in cectB_dataloader:
#             prototypes = [[] for _ in range(opt['classes_num'])]
#             # attention = main_models.MultiHeadAttention(1, opt['classifier_input_dim'], opt['classifier_input_dim'], opt['classifier_input_dim'], dropout=0)
#             # attention = attention.to(device)
#             for i in range(len(X_t)):
#                 prototypes[int(Y_t[i])].append(np.array(encoder(X_t[i].unsqueeze(0).to(device)).detach().cpu()))
#             prototypes = torch.Tensor(prototypes).mean(1)
#             prototypes.to(device)


#             data=data.to(device)
#             labels=labels.to(device)

#             optimizer.zero_grad()

#             encoder_vectors = encoder(data)
#             y_pred=classifier(encoder_vectors)
#             y_prototypes = main_models.prototype_loss(encoder_vectors,prototypes.to(device),labels)


#             # num_query = encoder_vectors.shape[0]
#             # proto = prototypes.squeeze(1).unsqueeze(0).repeat([num_query, 1, 1])  # NK x N x d[batch_size,23,256]
#             # proto = proto.to(device)
#             # query = encoder_vectors.unsqueeze(1) # [batch_size,1,256]
#             # combined = torch.cat([proto, query], 1) # Nk x (N + 1) x d, batch_size = NK

#             # combined, _, att = attention(combined, combined, combined)
#             # refined_support, refined_query = combined.split(opt['classes_num'], 1)
#             # refined_query = refined_query.squeeze(1)
#             # y_pred=classifier(refined_query)


#             # att_label_basis = []
#             # for i in range(opt['classes_num']):
#             #     temp = torch.eye(opt['classes_num'] + 1)
#             #     temp[i, i] = 0.5
#             #     temp[-1, -1] = 0.5
#             #     temp[i, -1] = 0.5
#             #     temp[-1, i] = 0.5
#             #     att_label_basis.append(temp)

#             # att_label = torch.zeros(labels.shape[0], opt['classes_num'] + 1, opt['classes_num'] + 1)
#             # for i in range(att_label.shape[0]):
#             #     att_label[i,:] = att_label_basis[labels[i].item()]

#             # att_label = att_label.to(device)

#             # loss_att = F.kl_div(att.view(-1, opt['classes_num'] + 1), att_label.view(-1, opt['classes_num'] + 1))
#             loss=loss_fn(y_pred,labels)
#             loss_proto=loss_fn(y_prototypes,labels)
#             loss = loss + loss_proto
#             # loss = loss + 400 * loss_att

#             # print(loss)
#             # print(loss_att)

#             loss.backward()
#             optimizer.step()

#         acc=0
#         for data,labels in cectA_dataloader:
#             data=data.to(device)
#             labels=labels.to(device)
#             encoder_vectors = encoder(data)
#             # y_prototypes = main_models.prototype_loss(encoder_vectors,prototypes.to(device),labels)
#             y_test_pred=classifier(encoder_vectors)
#             acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()

#         accuracy=round(acc / float(len(cectA_dataloader)), 3)

#         print("step1----Epoch %d/%d  accuracy: %.3f "%(epoch+1,opt['n_epoches_1'],accuracy))


# # final testing
# acc = 0
# for data, labels in test_dataloader:
#     data = data.to(device)
#     labels = labels.to(device)
#     y_test_pred = classifier(encoder(data))
#     acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

# accuracy = round(acc / float(len(test_dataloader)), 3)

# print("accuracy: %.3f " % accuracy)

encoder1_S.load_state_dict(torch.load('results/encoder1S_stage3.pt'))
encoder1_T.load_state_dict(torch.load('results/encoder1T_stage3.pt'))
encoder2.load_state_dict(torch.load('results/encoder2_stage3.pt'))
classifier.load_state_dict(torch.load('results/classifier_stage3.pt'))

# testing
acc = 0
end = time.time()
print('time cost:%f' % (end - begin))
# for data, labels in test_dataloader:
#    data = data.to(device)
#    labels = labels.to(device)

#    _, encoder_data = encoder2(encoder1_T((data)))

#    y_test_pred = classifier(encoder_data)
#    acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

# accuracy = round(acc / float(len(test_dataloader)), 3)
# print("accuracy: %.3f " % accuracy)
