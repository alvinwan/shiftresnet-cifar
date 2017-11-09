# ShiftResNet

Train ResNet with shift operations on CIFAR10 using PyTorch. This uses the [original resnet codebase](https://github.com/kuangliu/pytorch-cifar.git) written by Kuang Liu. In this codebase, we replace 3x3 convolutional layers with a conv-shift-conv--a 1x1 convolutional layer, a set of shift operations, and a second 1x1 convolutional layer. From Liu, this repository boasts:

- Built-in data loading and augmentation, very nice!
- Training is fast, maybe even a little bit faster.
- Very memory efficient!

## Getting Started

1. Clone the repository

```
git clone git@github.com:alvinwan/shiftresnet-cifar.git
```

2. Compile the Shift Layer implementation in C.
```
cd shiftresnet-cifar/models/shiftnet_cuda_v2
make
cd ../../
```
3. Run
```
python main.py
```
## Accuracy

Below, we run experiments using ResNet101, varying expansion used for all conv-shift-conv layers in the neural network. Here, we consider 3x3 filters. `a` is the number of filters in the first set of 1x1 convolutional filters. `c` is the number of channels in our input. The original ResNet110 has accuracy ___.

| Expansion | a | Acc |
|-----------|---|-----
| 1 | c | 
| 2 | 2c |
| 3 | 3c |
| 4 | 4c |
| 5 | 5c |
| 6 | 6c |
| 7 | 7c |
| 8 | 8c |
| 9 | 9c |
