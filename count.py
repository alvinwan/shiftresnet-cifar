from models import ResNet20
from models import ShiftResNet20
from models import ResNet44
from models import ShiftResNet44
from models import ResNet56
from models import ShiftResNet56
from models import ResNet110
from models import ShiftResNet110
import numpy as np
import argparse

all_models = {
    'resnet20': ResNet20,
    'shiftresnet20': ShiftResNet20,
    'resnet44': ResNet44,
    'shiftresnet44': ShiftResNet44,
    'resnet56': ResNet56,
    'shiftresnet56': ShiftResNet56,
    'resnet110': ResNet110,
    'shiftresnet110': ShiftResNet110
}

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--arch', choices=all_models.keys(),
                    help='Architecture to count parameters for', default='shiftresnet110')
parser.add_argument('--expansion', type=int, default=1, help='expansion for shift layers')
args = parser.parse_args()

def count_params(net):
     return sum([np.prod(param.size()) for name, param in net.named_parameters()])

original = all_models[args.arch.replace('shift', '')]()
original_count = count_params(original)

cls = all_models[args.arch]
net = cls() if 'shift' not in args.arch else cls(expansion=args.expansion)
new_count = count_params(net)

print('Parameters: (new) %d (original) %d (reduction) %.2f' % (
      new_count, original_count, float(original_count) / new_count))
