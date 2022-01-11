import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.autograd import Variable
from sklearn.metrics import classification_report
import aitom_core.simulation.reconstruction.reconstruction__simple_convolution as recon
from gaussian import gaussian3D
from torch.nn import functional as F
from scipy.ndimage import zoom



class Conv_block_T(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block_T, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class decoder_general(nn.Module):
    def __init__(self, channel_list, name):

        super(decoder_general, self).__init__()
        self.u0 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=channel_list[0],
                out_channels=channel_list[1],
                kernel_size=2,
                stride=2,
            ),
            Conv_block(channel_list[1], channel_list[2]),
        )
        #   2  ->   [128,8,8]
        #   1  ->   [256,4,4]
        self.u1 = nn.Sequential(
            Conv_block_T(channel_list[2] * 2, channel_list[3]),
            Conv_block(channel_list[3], channel_list[4]),
            Conv_block(channel_list[4], channel_list[5]),
        )
        #   5  ->   [128,16,16]
        self.u2 = nn.Sequential(
            Conv_block_T(channel_list[5] * 2, channel_list[6]),
            Conv_block(channel_list[6], channel_list[7]),
            Conv_block(channel_list[7], channel_list[8]),
        )
        #   8  ->   [64,32,32]
        self.u3 = nn.Sequential(
            Conv_block(channel_list[8] * 2, channel_list[9]),
            Conv_block(channel_list[9], channel_list[11]),
        )
        self.name = name

    def forward(self, store):
        x = self.u0(store["encode_" + self.name])
        x = torch.cat([x, store["2"]], dim=1)
        x = self.u1(x)
        x = torch.cat([x, store["1"]], dim=1)
        x = self.u2(x)
        x = torch.cat([x, store["0"]], dim=1)
        # print(x.shape)
        x = self.u3(x)
        return x


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class deform(nn.Module):
    def __init__(
        self,
        shape,
        channels=1,
        kernel_size=11,
        sigma=1,
        stride=1,
        padding=5,
        dim=2,
        device="cuda",
    ):

        super(deform, self).__init__()
        padding = (kernel_size - 1) // 2
        kernel = torch.tensor(gaussian3D(sigma, 0, 1, kernel_size)).to(device)
        self.device = device
        kernel = kernel.view(1, 1, *kernel.size()).to(device)
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).to(device).float()
        
        self.weight = kernel
        self.groups = channels
        self.conv = F.conv3d
        self.padding = padding
        self.stride = stride
        self.shape = shape
        self.x, self.y, self.z = torch.meshgrid(
            torch.arange(shape[2]), torch.arange(shape[2]), torch.arange(shape[2])
        )

        self.x = self.x.to(device)
        self.x.requires_grad = False
        self.y = self.y.to(device)
        self.y.requires_grad = False

        self.z = self.z.to(device)
        self.z.requires_grad = False

        channel_list_ns = [128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 32, 3]
        # channel_list_ns = [64, 64, 64, 64, 64, 64, 64, 32, 32, 32, 16, 3]
        self.dec = nn.Sequential(
            decoder_general(channel_list_ns, name="ds"), nn.Sigmoid()
        )

    def forward(self, bottleneck, inputs, alphas, mask, ds_lim):
        device = self.device
        self.x = self.x.repeat(inputs.shape[0], 1, 1, 1, 1)
        self.y = self.y.repeat(inputs.shape[0], 1, 1, 1, 1)
        self.z = self.z.repeat(inputs.shape[0], 1, 1, 1, 1)
        d = self.dec(bottleneck)
        d = (d - 0.5) * 2.0 * ds_lim

        dx = d[:, 0, :, :, :].unsqueeze(1)
        dy = d[:, 1, :, :, :].unsqueeze(1)
        dz = d[:, 2, :, :, :].unsqueeze(1)

        dx = self.conv(
            dx.to(device),
            weight=self.weight,
            groups=self.groups,
            stride=self.stride,
            padding=self.padding,
        )
        dy = self.conv(
            dy.to(device),
            weight=self.weight,
            groups=self.groups,
            stride=self.stride,
            padding=self.padding,
        )
        dz = self.conv(
            dz.to(device),
            weight=self.weight,
            groups=self.groups,
            stride=self.stride,
            padding=self.padding,
        )

        for i in range(len(alphas)):
            dx[i] *= alphas[i]
            dy[i] *= alphas[i]
            dz[i] *= alphas[i]
        x = ((((self.x) / (self.x.shape[2] - 1) - 0.5) * 2) + dx).squeeze(1)
        y = ((((self.y) / (self.y.shape[2] - 1) - 0.5) * 2) + dy).squeeze(1)
        z = ((((self.z) / (self.z.shape[2] - 1) - 0.5) * 2) + dz).squeeze(1)

        grid = torch.cat([z.unsqueeze(4), y.unsqueeze(4), x.unsqueeze(4)], -1)

        return (
            torch.nn.functional.grid_sample(
                inputs, grid, align_corners=True, padding_mode="reflection"
            ),
            torch.nn.functional.grid_sample(
                mask.unsqueeze(1), grid, align_corners=True, padding_mode="reflection"
            ),
        )


def create_set(batch_X, batch_y, r, test_batch_X, test_batch_y):

    batch_y = np.squeeze(batch_y)
    batch_lab = np.array([1] * len(batch_X))
    batch_lab_y = np.array([0] * len(batch_X))
    create_X = np.zeros((batch_X.shape))
    create_Y = np.zeros((batch_y.shape))
    for k in range(len(batch_y)):
        i = batch_y[k]
        create_X[k, :, :, :, :] = test_batch_X[
            int(np.random.randint(r[i], r[i] + (r[i + 1] - r[i]) // 10)), :, :, :, :
        ]
        create_Y[k] = int(i)

    create_X = torch.tensor(create_X).cuda()
    

    ar_in = torch.cat([batch_X, create_X], dim=0).float()

    ar_lab = torch.tensor(np.concatenate((batch_lab, batch_lab_y), axis=0)).cuda()

    labels = np.int64(
        np.concatenate((batch_y.detach().cpu().numpy(), create_Y), axis=0)
    )
    return ar_in, ar_lab, labels


def get_batch(DMs, DM_masks, n_batch, rangeParams, pool, re=False):
    batch_X, batch_y, batch_mask = recon.data_generator(
        DMs, DM_masks, n_batch, rangeParams, pool
    )

    if re:
        batch_X = zoom(batch_X, (1, 1, 0.8, 0.8, 0.8), order=1)
        batch_mask = zoom(batch_mask, (1, 0.8, 0.8, 0.8), order=1)
        batch_mask[batch_mask >= 0.5] = 1.0
        batch_mask[batch_mask < 0.5] = 0.0

    for batch in range(len(batch_X)):

        batch_X[batch, :] = batch_X[batch] - np.min(batch_X[batch])
        batch_X[batch, :] /= np.max(batch_X[batch])

    batch_mask_torch = torch.from_numpy(batch_mask).float().cuda()

    input = torch.from_numpy(batch_X)
    input = Variable(input.float()).cuda()

    target = Variable(torch.tensor(batch_y).squeeze(1)).cuda()

    return input, target, batch_mask_torch


def test(
    net,
    device,
    test_loader,
    use_tqdm=False,
    segmentation=False,
    classification=False,
    log=True,
    out_size=1,
):
    net.eval()

    t_correct = []
    t_paccs = []

    t_IoUs = []
    y_pred = []
    y_true = []
    loss_test = 0
    with torch.no_grad():
        for input, target, mask in tqdm(test_loader, disable=not use_tqdm):

            input, target, mask = (
                input.to(device),
                target.squeeze(1).to(device),
                mask.to(device),
            )
            for batch in range(len(input)):
                input[batch] = input[batch] - torch.min(input[batch])
                input[batch] = input[batch] / torch.max(input[batch])
            if segmentation:
                seg = net.forward(input.float())
                loss_test += float(dice_loss(seg, mask.unsqueeze(1).float()))
                pred_mask = (seg.squeeze() > 0.5).float().cpu().numpy()
                y_mask = mask
                pixel_acc = np.mean(1.0 * pred_mask == y_mask.cpu().numpy())
                t_paccs.append(pixel_acc)

                test_IoU = 0
                I = np.sum(np.logical_and(pred_mask, y_mask.detach().cpu().numpy()))
                U = np.sum(np.logical_or(pred_mask, y_mask.detach().cpu().numpy()))
                test_IoU = I / U
                t_IoUs.append(test_IoU)

            elif classification:
                if out_size == 2:
                    _, output = net.forward(input.float())
                else:
                    output = net.forward(input.float())
                loss_test += float(nn.CrossEntropyLoss()(output, target))
                output = torch.softmax(output, dim=1)

                predicted = torch.argmax(output, dim=1)
                pred = predicted.data.cpu().numpy()

                for i in range(len(output)):
                    y_pred.append(pred[i])
                    y_true.append(float(target[i]))
                t_correct.append(np.sum(pred == target.cpu().numpy()) / len(pred))

        if segmentation:
            if log:
                print("===================================================")
                print(
                    "Test [Pacc, IoU]: = [{:.4f},{:.4f}]".format(
                        np.mean(t_paccs),
                        np.mean(t_IoUs),
                    )
                )
            return loss_test, np.mean(t_IoUs)
        elif classification:
            if log:
                print("===================================================")
                print(
                    "Test [Accuracy]: = [{:.4f}]".format(
                        np.mean(t_correct),
                    )
                )
                print("Classification Report")
                print(classification_report(y_true, y_pred))
            return loss_test, np.mean(t_correct)


def vis(dm):
    plt.ion()
    plt.figure()
    img = np.reshape(dm, [5, 8, 40, 40])
    img = np.transpose(img, [0, 2, 1, 3])
    img = np.reshape(img, [200, 320])
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    plt.imshow(img, cmap="gray")
    plt.show()


def normalize(im, im3d):
    return (im - np.min(im3d)) / (np.max(im3d) - np.min(im3d))


def save(im3d, id, file):
    Image.fromarray((normalize(im3d[id], im3d) * 255).astype(np.uint8)).save(file)


def save_2(dm, file):
    img = np.reshape(dm, [40, 40, 40])
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.concatenate([img, np.zeros([9, 40, 40])])
    img = np.reshape(img, [7, 7, 40, 40])
    img = np.transpose(img, [0, 2, 1, 3])
    img = np.reshape(img, [280, 280])
    Image.fromarray((img * 255).astype(np.uint8)).save(file)


def vis_2(dm):
    plt.ion()
    plt.figure()
    img = np.reshape(dm, [40, 40, 40])
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.concatenate([img, np.zeros([9, 40, 40])])
    img = np.reshape(img, [7, 7, 40, 40])
    img = np.transpose(img, [0, 2, 1, 3])
    img = np.reshape(img, [280, 280])
    plt.imshow(img, cmap="gray")
    plt.show()