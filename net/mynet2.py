""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class Down1(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, size):
        super().__init__()

        self.max = nn.MaxPool2d(2)
        self.asp = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=1, padding=0, dilation=size // 2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.fc = nn.Conv2d(in_channels * 2, in_channels, padding=0, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.max(x)
        x2 = self.asp(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.fc(out)
        return self.conv(out)

class ConnectBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.att = ChannelAttention(64 + channels)
        self.conv = nn.Conv2d(57, 64, padding=0, kernel_size=3, stride=1)
        self.fc = nn.Conv2d(64 + channels , channels, padding=0, kernel_size=1, stride=1)

    def forward(self, conv, vit):

        vit = vit.reshape(16, 57, 16, 16)
        vit = self.conv(vit)
        out = torch.cat([conv, vit], dim=1)
        out = self.att(out)
        out = self.fc(out)
        return out

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, int(channels/reduction), kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(int(channels/reduction), channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class Attention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(Attention, self).__init__()

        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out1 = self.channel_att(x)
        out2 = self.spatial_att(out1)
        return out2


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, vit, batch):
        super(UNet, self).__init__()
        self.batch = batch
        self.vit = vit
        self.inc = (DoubleConv(n_channels, 64))

        self.down1 = (Down1(64, 128, 224))
        self.down2 = (Down1(128, 256, 112))
        self.down3 = (Down1(256, 512, 56))
        self.down4 = (Down1(512, 1024, 28))

        self.connect = ConnectBlock(1024)

        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))

        self.outc = (OutConv(64, n_classes))

        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        vit = self.vit(x)
        x1 = self.inc(y)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.connect(x5, vit)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
