from models import ResNet20
from models import ShiftResNet20
from models import ResNet56
from models import ShiftResNet56
from models import ResNet110
from models import ShiftResNet110
import torch
from torch.autograd import Variable
import numpy as np
import argparse

all_models = {
    'resnet20': ResNet20,
    'shiftresnet20': ShiftResNet20,
    'resnet56': ResNet56,
    'shiftresnet56': ShiftResNet56,
    'resnet110': ResNet110,
    'shiftresnet110': ShiftResNet110,
}

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--arch', choices=all_models.keys(),
                    help='Architecture to count parameters for', default='shiftresnet110')
parser.add_argument('--expansion', type=int, default=1, help='expansion for shift layers')
parser.add_argument('--reduction', type=float, default=1, help='reduction for resnet')
parser.add_argument('--reduction-mode', choices=('block', 'net', 'depthwise', 'shuffle', 'mobile'), help='"block" reduces inner representation for BasicBlock, "net" reduces for all layers', default='net')
args = parser.parse_args()

def count_params(net):
     return sum([np.prod(param.size()) for name, param in net.named_parameters()])

def count_flops(net):
     """Approximately count number of FLOPs"""
     dummy = Variable(torch.randn(1, 3, 32, 32)).cuda()  # size is specific to cifar10, cifar100!
     net.cuda().forward(dummy)
     return net.flops()

original = all_models[args.arch.replace('shift', '')]()
original_count = count_params(original)
original_flops = count_flops(original)

cls = all_models[args.arch]

assert 'shift' not in args.arch or args.reduction == 1, \
    'Only default resnet supports reductions'
if args.reduction != 1:
    print('==> %s with reduction %.2f' % (args.arch, args.reduction))
    net = cls(reduction=args.reduction, reduction_mode=args.reduction_mode)
else:
    net = cls() if 'shift' not in args.arch else cls(expansion=args.expansion)
new_count = count_params(net)
new_flops = count_flops(net)

print('Parameters: (new) %d (original) %d (reduction) %.2f' % (
      new_count, original_count, float(original_count) / new_count))
print('FLOPs: (new) %d (original) %d (reduction) %.2f' % (
      new_flops, original_flops, float(original_flops) / new_flops))
