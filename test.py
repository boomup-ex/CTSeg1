import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import TrainDataset,EvalDataset
from net.mynet import UNet
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from loss.dice import MultiClassDiceLoss
import torch
import torchvision

from torchstat import stat

model = torchvision.models.vgg16(pretrained = False)
device = torch.device('cpu')
model.to(device)



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(3,2).cuda()
    weights_dict = torch.load("./res_n2_dice0407_iou0737.pth", map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)

    print('---------------success loading')

    Evaldataset = EvalDataset('./val/')
    Evalloader = DataLoader(dataset=Evaldataset, batch_size=1, shuffle=True, num_workers=0)

    model = model.to(device)

    loss1 = MultiClassDiceLoss()
    loss2 = nn.CrossEntropyLoss()
    logging.basicConfig(filename='example3.log', level=logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')
    loss_each = 0.0
    num = 0
    for i, (origin, mask) in enumerate(Evalloader):
        #print(mask)
        org = origin.to(device)
        mask1 = mask.to(device)
        out = model(org)
        my_loss = 0.5*loss1(out, mask1) + 0.5*loss2(out, mask1)
        loss = my_loss.item()
        print('loss = ' + str(my_loss))
        loss_each = loss_each + loss
        num = num + 1
    loss_avg = loss_each / num
    print('loss_avg = ' + str(loss_avg))



