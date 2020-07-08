# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, padding=1, activate='relu'):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding))
    layers.append(nn.BatchNorm2d(out_channels))
    if activate == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activate == 'sigmoid':
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def double_conv3x3(in_channels, out_channels, stride=1, padding=1, activate='relu'):
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride, padding=1, activate=activate),
        conv3x3(out_channels, out_channels, stride, padding=1, activate=activate))

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            double_conv3x3(in_channels, out_channels))

    def forward(self, x):
        return self.net(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()
        self.bilinear = bilinear
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.net = double_conv3x3(in_channels, out_channels)

    def forward(self, front, later):
        if self.bilinear:
            later = F.interpolate(later, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            later = self.conv_trans(later)
        h_diff = front.size()[2] - later.size()[2]
        w_diff = front.size()[3] - later.size()[3]
        later = F.pad(later, pad=(w_diff // 2, w_diff - w_diff // 2, h_diff // 2, h_diff - h_diff // 2),
                      mode='constant', value=0)
        x = torch.cat([front, later], dim=1)
        x = self.net(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(UNet, self).__init__()
        self.inconv = double_conv3x3(in_ch, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 512)
        self.up1 = UpSample(1024, 256)
        self.up2 = UpSample(512, 128)
        self.up3 = UpSample(256, 64)
        self.up4 = UpSample(128, 64)
        self.outconv = double_conv3x3(64, out_ch, activate='sigmoid')

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.outconv(x)

        return x