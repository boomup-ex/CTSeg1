import numpy as np
from torch.utils.data import Dataset
import os
import torch
import matplotlib.pyplot as plt
import cv2

def get_edge(img):
    filtered_image = cv2.bilateralFilter(img, d=4, sigmaColor=75, sigmaSpace=75)

    gray = filtered_image
    # 3.计算梯度
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)  # canny方法API要求不允许使用浮点数
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    # 4.Canny方法中包含非最大信号抑制和双阈值输出
    edge_output = cv2.Canny(xgrad, ygrad, 70, 180)  # 50是低阈值，150是高阈值

    return edge_output


class TrainDataset(Dataset):
    def __init__(self, path):
        super(TrainDataset, self).__init__()
        imgs = []
        print(path)
        list = os.listdir(str(path)+'origin/')
        for image in list:
            img1 = np.load(str(path) +'origin/'+ image).astype(np.float32)
            img2 = get_edge(img1).astype(np.float32)
            img3 = img1*img2
            data1 = np.expand_dims(img1, 0).astype(np.float32)
            data2 = np.expand_dims(img2, 0).astype(np.float32)
            data3 = np.expand_dims(img3, 0).astype(np.float32)
            data11 = np.concatenate((data1, data2, data3), axis=0)
            data22 = np.concatenate((data1, data1, data1), axis=0)
            data33 = np.load(str(path) + 'label/' + image).astype(np.float32)
            imgs.append([torch.tensor(data11, dtype=torch.float), torch.tensor(data22, dtype=torch.float),torch.tensor(data33,dtype=torch.long)])
        print(type(imgs))
        self.imgs = imgs

    def __getitem__(self, idx):
        return self.imgs[idx][0], self.imgs[idx][1], self.imgs[idx][2]

    def __len__(self):
        return len(self.imgs)

class EvalDataset(Dataset):
    def __init__(self, path):
        super(EvalDataset, self).__init__()
        imgs = []
        print(path)
        list = os.listdir(str(path)+'origin/')
        for image in list:
            img1 = np.load(str(path) +'origin/'+ image).astype(np.float32)
            img2 = get_edge(img1).astype(np.float32)
            img3 = img1*img2
            data1 = np.expand_dims(img1, 0).astype(np.float32)
            data2 = np.expand_dims(img2, 0).astype(np.float32)
            data3 = np.expand_dims(img3, 0).astype(np.float32)
            data11 = np.concatenate((data1, data2, data3), axis=0)
            data22 = np.concatenate((data1, data1, data1), axis=0)
            data33 = np.load(str(path) + 'label/' + image).astype(np.float32)
            imgs.append([torch.tensor(data11, dtype=torch.float), torch.tensor(data22, dtype=torch.float),torch.tensor(data33,dtype=torch.long)])
        print(type(imgs))
        self.imgs = imgs

    def __getitem__(self, idx):
        return self.imgs[idx][0], self.imgs[idx][1], self.imgs[idx][2]

    def __len__(self):
        return len(self.imgs)