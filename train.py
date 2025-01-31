import torch
import torch.nn as nn
from dataset import TrainDataset, EvalDataset
from net.mynet2 import UNet
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from loss.dice import MultiClassDiceLoss
from loss.losses_imbalance import IOULoss
from loss.MENLoss import MENloss
from net.lightvit import LightViT
import pandas as pd

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Vmodel = LightViT(patch_size=8, embed_dims=[64, 128, 256], num_layers=[2, 6, 6],
                      num_heads=[2, 4, 8, ], mlp_ratios=[8, 4, 4], num_tokens=8).cuda()
    ckpt = torch.load('./lightvit.ckpt', map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    elif 'model' in ckpt:
        ckpt = ckpt['model']
    Vmodel.load_state_dict(ckpt, strict=True)
    Vmodel.neck = nn.Sequential()
    Vmodel.head = nn.Sequential()
    Vmodel.norm = nn.Sequential()
    # Vmodel.stages[1] = nn.Sequential()
    # Vmodel.stages[2] = nn.Sequential()
    Vmodel = Vmodel.to(device)
    for param in Vmodel.parameters():
        param.requires_grad = False
    model = UNet(3, 4, Vmodel, 16).to(device)
    weights_dict = torch.load("./CTSeg_dice0812_iou1397.pth", map_location='cpu')
        #print(weights_dict)
    model.load_state_dict(weights_dict, strict=True)

    print('==============OK==========')

    Traindataset = TrainDataset('../data/train/')
    Trainloader = DataLoader(dataset=Traindataset, batch_size=16, shuffle=True, num_workers=0)
    Evaldataset = EvalDataset('../data/val/')
    Evalloader = DataLoader(dataset=Evaldataset, batch_size=16, shuffle=True, num_workers=0)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    loss1 = nn.CrossEntropyLoss()
    loss2 = MultiClassDiceLoss()
    loss3 = IOULoss(1)
    loss4 = MENloss()
    best_loss = 0.0

    torch.manual_seed(2023)
    np.random.seed(2023)
    random.seed(2023)
    logging.basicConfig(filename='example3.log', level=logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')
    epoch = 300
    best_dice = 10.0
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    train_dice = []
    train_iou = []
    eval_dice = []
    eval_iou = []
    for e in range(epoch):
        model.train()
        loss_each = 0.0
        num = 0.0
        step = 0
        Diceloss = 0.0
        IOUloss = 0.0
        loss_avg = 0
        for i, (origin1, origin2, mask) in enumerate(Trainloader):
            org1 = origin1.to(device)
            org2 = origin2.to(device)
            mask = mask.to(device)
            out = model(org1, org2)
            my_loss1 = loss1(out, mask)
            my_loss2 = loss2(out, mask)
            my_loss3 = loss3(out, mask)
            my_loss4 = loss4(out)
            my_loss = my_loss1 + my_loss2 + my_loss3 * 1.5 + my_loss4 * 0.5
            my_loss.backward()
            optimizer.step()
            scheduler.step()
            loss_all = my_loss.item()
            CEloss = my_loss1.item()
            Diceloss = my_loss2.item()
            IOUloss = my_loss3.item()
            optimizer.zero_grad()
            loss_each = loss_each + loss_all
            num = num + 1
            loss_avg = loss_each / num
            step = step + 16
            with open("./train_loss.txt", 'w') as train_los:
                train_los.write(str(Diceloss))
            print((
                'epoch = {0:8d}/{1:8d} ,step = {2:8d},loss={3:8f}, CE loss={4:8f},Dice loss={5:8f} IOU loss={6:8f} loss_avg = {7:8f}  '.format(
                    e, epoch,
                    step,
                    loss_all, CEloss, Diceloss, IOUloss, loss_avg)))
            logging.debug((
                'epoch = {0:8d}/{1:8d} ,step = {2:8d},loss={3:8f}, CE loss={4:8f},Dice loss={5:8f} IOU loss={6:8f} loss_avg = {7:8f}  '.format(
                    e, epoch,
                    step,
                    loss_all, CEloss, Diceloss, IOUloss, loss_avg)))
        # -------------eval------------
        train_dice.append(Diceloss)
        train_iou.append(IOUloss)
        # -------------eval------------
        loss_sum = 0.
        CE_avg = 0.
        Dice_avg = 0.
        IOU_avg = 0.
        model.eval()
        for i, (origin1, origin2, mask) in enumerate(Evalloader):
            org1 = origin1.to(device)
            org2 = origin2.to(device)
            mask1 = mask.to(device)
            out = model(org1, org2)
            my_loss1 = loss1(out, mask1)
            my_loss2 = loss2(out, mask1)
            my_loss3 = loss3(out, mask1)
            CE_loss = my_loss1.item()
            DICE_loss = my_loss2.item()
            IOUloss = my_loss3.item()
            loss_sum = loss_sum + CE_loss + DICE_loss + IOUloss
            CE_avg = CE_avg + my_loss1.item()
            Dice_avg = Dice_avg + my_loss2.item()
            IOU_avg = IOU_avg + my_loss3.item()
        loss_avg = loss_sum / 2
        CE_avg = CE_avg / 2
        Dice_avg = Dice_avg / 2
        IOU_avg = IOU_avg / 2
        eval_dice.append(Dice_avg)
        eval_iou.append(IOU_avg)

        print(('Eval: CE loss = {0:8f} DICE loss = {1:8f} IOU loss = {2:8f}  loss_avg = {3:8f} epoch:{4:8d}'.format(
            CE_avg, Dice_avg,
            loss_avg, IOU_avg, e)))
        logging.debug((
                          'Eval: CE loss = {0:8f} DICE loss = {1:8f} IOU loss = {2:8f}  loss_avg = {3:8f} epoch:{4:8d}'.format(
                              CE_avg, Dice_avg,
                              loss_avg, IOU_avg, e)))
        with open('./ex4.txt', 'a') as f:
            f.writelines(
                'epoch:' + str(e) + 'Eval: IOU loss = ' + str(float(IOU_avg)) + ' CE_loss = ' + str(
                    float(CE_avg)) + ' DICE_loss = ' + str(
                    float(Dice_avg)) + '\n')

        if best_dice > Dice_avg:
            best_dice = Dice_avg
            with open('./best.txt', 'a') as f:
                f.writelines(
                    'epoch:' + str(e) + 'Eval: IOU loss = ' + str(float(IOU_avg)) + ' CE_loss = ' + str(
                        float(CE_avg)) + ' DICE_loss = ' + str(
                        float(Dice_avg)) + '\n')
            torch.save(model.state_dict(), "./param/model_p" + str(e) + ".pth")
    with open("./train_dice.txt", 'w') as train_d:
        train_d.write(str(train_dice))

    with open("./train_iou.txt", 'w') as train_i:
        train_i.write(str(train_iou))

    with open("./eval_dice.txt", 'w') as eval_d:
        eval_d.write(str(eval_dice))

    with open("./eval_iou.txt", 'w') as eval_i:
        eval_i.write(str(eval_iou))



