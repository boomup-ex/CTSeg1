import torch
import torch.nn as nn


class MENloss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, f1, f2):
        return torch.mean(torch.sigmoid_(torch.exp_(f1)*f1 + torch.exp_(f2)*f2))

