import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from utils import *
from .layers import Attention


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class CBAA(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(CBAA, self).__init__()
        self.conv1 = SeparableConv2d(in_channels,out_channels,3,1)
        self.batchnormal = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()
        self.attention=Attention(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnormal(x)
        x = self.relu(x)
        x = self.attention(x)
        return x

class att(nn.Module):
    def __init__(self,M=4):
        super(att, self).__init__()
        self.net1 = nn.Sequential()
        for m in range(1, M + 1):
            if m==1:
                self.net1.add_module('cbaa%d'%m,CBAA(3,m*32))
            elif m==3:
                self.net1.add_module('cbaa%d'%m,CBAA((m-1)*32,m*32))
            else:
                self.net1.add_module('cbaa%d' % m, CBAA((m - 1) * 32, m*32))
        self.conv=nn.Conv2d(32*M,576,1,1)
        self.batchnormal = nn.BatchNorm2d(576)
        self.relu = nn.ReLU()
        self.globalaverage=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.net1(x)
        x = self.conv(x)
        x = self.batchnormal(x)
        x1 = self.relu(x)
        out = self.globalaverage(x1)
        out = out.view(out.size(0), -1)
        return out

if __name__ == '__main__':
    input=torch.randn(2,3,64,64)
    model=att(4)

    out=model(input)
    print(out.shape)

