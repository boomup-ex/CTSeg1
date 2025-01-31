""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
#from Involution import Inv2d
from get_sobel import run_sobel, get_sobel



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
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
        self.asp1 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=1, padding=0, dilation=size // 2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.fc = nn.Conv2d(in_channels * 2, in_channels, padding=0, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.max(x)
        x2 = self.asp(x)
        x3 = self.asp1(x.transpose(2, 3))
        out = torch.cat([x1, x2*x3], dim=1)
        #out = torch.cat([x1, x2], dim=1)
        out = self.fc(out)
        return self.conv(out)

class Down2(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, aspp, size):
        super().__init__()

        self.max = nn.MaxPool2d(2)
        self.aspp = aspp
        self.asp = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=1, padding=0, dilation=size // 2)
        self.asp1 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=1, padding=0, dilation=size // 2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.fc = nn.Conv2d(in_channels * 3, in_channels, padding=0, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.max(x)
        x2 = self.asp(x)
        x3 = self.asp1(x.transpose(2, 3))
        x4 = self.aspp(x)
        out = torch.cat([x1, x2*x3, x4], dim=1)
        #out = torch.cat([x1, x2], dim=1)
        out = self.fc(out)
        return self.conv(out)


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

class PoolConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(PoolConv, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class MultiScaleConvBlock(nn.Module):
    def __init__(self, channels, channel_att, atrous_rates):
        super(MultiScaleConvBlock, self).__init__()
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.att = channel_att
        self.m1 = PoolConv(channels, channels)
        self.m2 = nn.Conv2d(channels, channels, kernel_size=3, padding=rate1, dilation=rate1)
        self.m3 = nn.Conv2d(channels, channels, kernel_size=3, padding=rate2, dilation=rate2)
        self.m4 = nn.Conv2d(channels, channels, kernel_size=3, padding=rate3, dilation=rate3)
        self.end = nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.float()
        x1 = self.m1(x)
        x2 = self.m2(x)
        x3 = self.m3(x)
        x4 = self.m4(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.att(out)
        out = self.end(out)
        return out

class FusionBlock(nn.Module):
    def __init__(self, channels, channel_att):
        super(FusionBlock, self).__init__()
        self.att = channel_att
        self.end = nn.Conv2d(channels*3, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        out = torch.cat([x2, x1], dim=1)
        out = self.att(out)
        out = self.end(out)
        return out

class ChannelAttention1(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention1, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, int(channels / reduction), kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(int(channels / reduction), channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = x * out
        return out + self.fc3(x)

class NonlocalAttention(nn.Module):
    def __init__(self, size, channels):
        super(NonlocalAttention, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((size, size))
        self.sigmoid = nn.Sigmoid()
        self.outc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = x - self.avg(x)
        x2 = x - self.avg(x.transpose(2, 3))
        x3 = self.sigmoid(x1 * x2)
        x4 = x3 + x
        out = self.outc(x4)

        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.ME1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, stride=1, padding=0)
        self.ME2 = nn.Conv2d(n_classes, n_classes, kernel_size=1, stride=1, padding=0)

        self.att4 = ChannelAttention(64 * 3)
        self.att3 = ChannelAttention(128 * 3)
        self.att2 = ChannelAttention(256 * 3)
        self.att1 = ChannelAttention(512 * 3)

        self.attt4 = ChannelAttention(1024 * 4)

        self.inc = (DoubleConv(n_channels, 64))

        self.mid_att = Attention(1024)

#         self.multi1 = MultiScaleConvBlock(64, channel_att=self.attt1, atrous_rates=[12, 24, 36])
#         self.multi2 = MultiScaleConvBlock(128, self.attt2, [12, 24, 36])  # [12, 24, 36]
#         self.multi3 = MultiScaleConvBlock(256, self.attt3, [8, 16, 24])  # [8, 16, 24]
#         self.multi4 = MultiScaleConvBlock(512, self.attt4, [4, 8, 12])  # [4, 8, 12]

        self.fu1 = FusionBlock(512, self.att1)
        self.fu2 = FusionBlock(256, self.att2)
        self.fu3 = FusionBlock(128, self.att3)
        self.fu4 = FusionBlock(64, self.att4)


        self.down1 = (Down1(64, 128, 224))
        self.down2 = (Down1(128, 256, 112))
        self.down3 = (Down1(256, 512, 56))
        self.down4 = (Down1(512, 1024, 28))

        #---------------------------Up----------------------------


        self.CA1 = ChannelAttention1(1024)
        self.NA1 = NonlocalAttention(14, 1024)
        self.CA2 = ChannelAttention(512)
        self.NA2 = NonlocalAttention(28, 512)
        self.CA3 = ChannelAttention(256)
        self.NA3 = NonlocalAttention(56, 256)
        self.CA4 = ChannelAttention(128)
        self.NA4 = NonlocalAttention(112, 128)
        self.CA5 = ChannelAttention(64)
        self.NA5 = NonlocalAttention(224, 64)

        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))

        #---------------------------Output----------------------------
        self.outc = (OutConv(64, n_classes))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.mid_att(x5)

        #------------------ feature fusion--------------------

        fu1 = self.fu1(x5, x4)
        fu2 = self.fu2(x4, x3)
        fu3 = self.fu3(x3, x2)
        fu4 = self.fu4(x2, x1)

        #-----------------------------------------------------

        x = self.up1(x5, fu1)
        x = self.CA2(x)
        x = self.NA2(x)
        x = self.up2(x, fu2)
        x = self.CA3(x)
        x = self.NA3(x)
        x = self.up3(x, fu3)
        x = self.CA4(x)
        x = self.NA4(x)
        x = self.up4(x, fu4)
        x = self.CA5(x)
        x = self.NA5(x)

        logits = self.outc(x)
        f1 = self.ME1(self.sig(logits))
        f2 = self.ME2(self.sig(logits))
        return logits, f1, f2
