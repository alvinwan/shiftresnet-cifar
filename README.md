# ShiftResNet

Train ResNet with shift operations on CIFAR10 using PyTorch. This uses the [original resnet codebase](https://github.com/kuangliu/pytorch-cifar.git) written by Kuang Liu. In this codebase, we replace 3x3 convolutional layers with a conv-shift-conv--a 1x1 convolutional layer, a set of shift operations, and a second 1x1 convolutional layer. From Liu, this repository boasts:

- Built-in data loading and augmentation, very nice!
- Training is fast, maybe even a little bit faster.
- Very memory efficient!

## Getting Started

1. Clone the repository

	git clone git@github.com:alvinwan/shiftresnet-cifar.git

2. Compile the Shift Layer implementation in C.

	cd shiftresnet-cifar/models/shiftnet_cuda_v2
        make
        cd ../../

3. Run

	python main.py

## Accuracy

Below, we run experiments using ResNet101, varying expansion used for all conv-shift-conv layers in the neural network.

| Expansion | Acc |
|-----------|-----|
| 1 | |
| 2 | |
| 3 | |
| 4 | |
| 5 | |
| 6 | |
| 7 | |
| 8 | |
| 9 | |

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,150)`
- `0.01` for epoch `[150,250)`
- `0.001` for epoch `[250,350)`

Resume the training with `python main.py --resume --lr=0.01`
