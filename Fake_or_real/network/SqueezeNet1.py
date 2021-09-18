import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import torch.optim as optim
import math

class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class squeezeNet1(nn.Module):
    def __init__(self):
        super(squeezeNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2) # 16
        self.fire1 = fire(96, 16, 64)
        self.fire2 = fire(128, 16, 64)
        self.fire3 = fire(128, 32, 128)
        self.fire4 = fire(256, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2) # 8

        self.fire5 = fire(256, 48, 192)
        self.fire6 = fire(384, 48, 192)
        self.fire7 = fire(384, 64, 256)
        self.fire8 = fire(512, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)  # 4

        self.conv2 = nn.Conv2d(512, 1000, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc=nn.Linear(1000,128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)

        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.conv2(x)
        # x1=x1.view(-1,2048)
        # x = self.avg_pool(x)
        # x0 = x.view(-1, 1000)
        # x = self.fc(x0)
        return x



if __name__ == '__main__':
    net = squeezeNet1()
    inp = Variable(torch.randn(10, 3, 64, 64))
    out = net.forward(inp)
    print(out.size())