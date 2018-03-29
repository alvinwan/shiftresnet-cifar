"""PyTorch implementation of DepthwiseResNet

ShiftResNet modifications written by Bichen Wu and Alvin Wan.

Reference:
[1] Bichen Wu, Alvin Wan, Xiangyu Yue, Peter Jin, Sicheng Zhao, Noah Golmant,
    Amir Gholaminejad, Joseph Gonzalez, Kurt Keutzer
    Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions.
    arXiv:1711.08141
"""

import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet


class DepthWiseWithSkipBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, reduction=1):
        super(DepthWiseWithSkipBlock, self).__init__()
        self.expansion = 1 / float(reduction)
        self.in_planes = in_planes
        self.mid_planes = mid_planes = int(self.expansion * out_planes)
        self.out_planes = out_planes

        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.depth = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, padding=1,
                               stride=1, bias=False, groups=mid_planes)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, bias=False, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (
        _, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.mid_planes * self.in_planes + out_h * out_w * self.mid_planes * self.out_planes
        flops += out_h * out_w * self.mid_planes * 9  # depth-wise convolution
        if len(self.shortcut) > 0:
            flops += self.in_planes * self.out_planes * out_h * out_w
        return flops

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = out.size()
        out = self.bn2(self.depth(out))
        out = self.bn3(self.conv3(out))
        self.out_nchw = out.size()
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def DepthwiseResNet20(reduction=1, num_classes=10):
    block = lambda in_planes, planes, stride: \
        DepthWiseWithSkipBlock(in_planes, planes, stride, reduction=reduction)
    return ResNet(block, [3, 3, 3], num_classes=num_classes)


def DepthwiseResNet56(reduction=1, num_classes=10):
    block = lambda in_planes, planes, stride: \
        DepthWiseWithSkipBlock(in_planes, planes, stride, reduction=reduction)
    return ResNet(block, [9, 9, 9], num_classes=num_classes)


def DepthwiseResNet110(reduction=1, num_classes=10):
    block = lambda in_planes, planes, stride: \
        DepthWiseWithSkipBlock(in_planes, planes, stride, reduction=reduction)
    return ResNet(block, [18, 18, 18], num_classes=num_classes)
