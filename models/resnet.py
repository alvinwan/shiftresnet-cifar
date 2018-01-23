'''PyTorch implementation of ShiftResNet

ShiftResNet modifications written by Bichen Wu and Alvin Wan, based
off of ResNet implementation by Kuang Liu.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from models.shiftnet_cuda_v2.nn import GenericShift_cuda
from models.shiftnet_cuda_v2.nn import Shift3x3_cuda

class ShiftConv(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, expansion=1):
        super(ShiftConv, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = mid_planes = int(out_planes * self.expansion)

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

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h*int_w*self.in_planes*self.mid_planes + out_h*out_w*self.mid_planes*self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes*self.out_planes*out_h*out_w
        return flops

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = x.size()
        x = F.relu(self.bn2(self.conv2(self.shift2(x))))
        self.out_nchw = x.size()
        x += shortcut
        return x


class DepthWiseBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, reduction=1):
        super(DepthWiseBlock, self).__init__()
        self.expansion = 1 / float(reduction)
        self.in_planes = in_planes
        self.mid_planes = mid_planes = int(self.expansion * out_planes)
        self.out_planes = out_planes

        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.depth = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, padding=1, stride=1, bias=False, groups=mid_planes)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, bias=False, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h*int_w*self.mid_planes*self.in_planes + out_h*out_w*self.mid_planes*self.out_planes
        flops += out_h*out_w*self.mid_planes*9  # depth-wise convolution
        if len(self.shortcut) > 0:
            flops += self.in_planes*self.out_planes*out_h*out_w
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


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, reduction=1):
        super(BasicBlock, self).__init__()
        self.expansion = 1 / float(reduction)
        self.in_planes = in_planes
        self.mid_planes = mid_planes = int(self.expansion * planes)
        self.out_planes = planes

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h*int_w*9*self.mid_planes*self.in_planes + out_h*out_w*9*self.mid_planes*self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes*self.out_planes*out_h*out_w
        return flops

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = out.size()
        out = self.bn2(self.conv2(out))
        self.out_nchw = out.size()
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, reduction=1, num_classes=10):
        super(ResNet, self).__init__()
        self.reduction = float(reduction) ** 0.5
        self.num_classes = num_classes
        self.in_planes = int(16 / self.reduction)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(32 / self.reduction), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(64 / self.reduction), num_blocks[2], stride=2)
        self.linear = nn.Linear(int(64 / self.reduction), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        planes = int(planes)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (out_h, out_w) = self.int_nchw, self.out_hw
        flops = 0
        for mod in (self.layer1, self.layer2, self.layer3):
            for layer in mod:
                flops += layer.flops()
        return int_h*int_w*9*self.in_planes + out_w*self.num_classes + flops

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = out.size()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        self.out_hw = out.size()
        out = self.linear(out)
        return out

def ResNetWrapper(num_blocks, reduction=1, reduction_mode='net', num_classes=10):
    if reduction_mode == 'block':
        block = lambda in_planes, planes, stride: \
            BasicBlock(in_planes, planes, stride, reduction=reduction)
        return ResNet(block, num_blocks, num_classes=num_classes)
    elif reduction_mode == 'depthwise':
        block = lambda in_planes, planes, stride: \
            DepthWiseBlock(in_planes, planes, stride, reduction=reduction)
        return ResNet(block, num_blocks, num_classes=num_classes)
    return ResNet(BasicBlock, num_blocks, num_classes=num_classes, reduction=reduction)

def ResNet20(reduction=1, reduction_mode='net', num_classes=10):
    return ResNetWrapper([3, 3, 3], reduction, reduction_mode, num_classes)

def ResNet32(reduction=1, reduction_mode='net', num_classes=10):
    return ResNetWrapper([5, 5, 5], reduction, reduction_mode, num_classes)

def ResNet44(reduction=1, reduction_mode='net',num_classes=10):
    return ResNetWrapper([7, 7, 7], reduction, reduction_mode, num_classes)

def ResNet56(reduction=1, reduction_mode='net',num_classes=10):
    return ResNetWrapper([9, 9, 9], reduction, reduction_mode, num_classes)

def ResNet110(reduction=1, reduction_mode='net',num_classes=10):
    return ResNetWrapper([18, 18, 18], reduction, reduction_mode, num_classes)

def ShiftResNet20(expansion=1, num_classes=10):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [3, 3, 3], num_classes=num_classes)

def ShiftResNet32(expansion=1, num_classes=10):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [5, 5, 5], num_classes=num_classes)

def ShiftResNet44(expansion=1, num_classes=10):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [7, 7, 7], num_classes=num_classes)

def ShiftResNet56(expansion=1, num_classes=10):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [9, 9, 9], num_classes=num_classes)

def ShiftResNet110(expansion=1, num_classes=10):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [18, 18, 18], num_classes=num_classes)
