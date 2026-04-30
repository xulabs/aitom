from networks.unet import UNet
from networks.VNet import VNet
import torch.nn as nn

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train"):
    if net_type == "unet" and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    return net

