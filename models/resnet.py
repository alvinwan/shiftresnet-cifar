'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from shiftnet_cuda_v2 import Shift3x3_cuda, GenericShift_cuda

class ShiftConv(nn.Module):
    expansion=9
    def __init__(self, in_planes, out_planes, stride=1, expansion=1):
        super(ShiftConv, self).__init__()
        self.expansion = expansion
        mid_planes = int(out_planes * self.expansion)

        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.shift2 = GenericShift_cuda(kernel_size=3, dilate_factor=1)
        self.conv2 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, bias=False, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                  in_planes, out_planes, kernel_size=1, stride=stride,
                  bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(self.shift2(x))))
        x += shortcut
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction=1):
        super(BasicBlock, self).__init__()
        self.expansion = 1 / reduction
        dim = int(self.expansion * planes)
        self.conv1 = nn.Conv2d(in_planes, dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(dim)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20(reduction=1):
    block = lambda in_planes, planes, stride: \
        BasicBlock(in_planes, planes, stride, reduction=reduction)
    return ResNet(block, [3, 3, 3])

def ResNet32(reduction=1):
    block = lambda in_planes, planes, stride: \
        BasicBlock(in_planes, planes, stride, reduction=reduction)
    return ResNet(block, [5, 5, 5])

def ResNet44(reduction=1):
    block = lambda in_planes, planes, stride: \
        BasicBlock(in_planes, planes, stride, reduction=reduction)
    return ResNet(block, [7 ,7, 7])

def ResNet56(reduction=1):
    block = lambda in_planes, planes, stride: \
        BasicBlock(in_planes, planes, stride, reduction=reduction)
    return ResNet(block, [9, 9, 9])

def ResNet110(reduction=1):
    block = lambda in_planes, planes, stride: \
        BasicBlock(in_planes, planes, stride, reduction=reduction)
    return ResNet(block, [18, 18, 18])

def ShiftResNet20(expansion=1):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [3, 3, 3])

def ShiftResNet32(expansion=1):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [5, 5, 5])

def ShiftResNet44(expansion=1):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [7, 7, 7])

def ShiftResNet56(expansion=1):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [9, 9, 9])

def ShiftResNet110(expansion=1):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [18, 18, 18])
