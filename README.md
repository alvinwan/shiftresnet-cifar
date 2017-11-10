# ShiftResNet

Train ResNet with shift operations on CIFAR10 using PyTorch. This uses the [original resnet codebase](https://github.com/kuangliu/pytorch-cifar.git) written by Kuang Liu. In this codebase, we replace 3x3 convolutional layers with a conv-shift-conv--a 1x1 convolutional layer, a set of shift operations, and a second 1x1 convolutional layer. From Liu, this repository boasts:

- Built-in data loading and augmentation, very nice!
- Training is fast, maybe even a little bit faster.
- Very memory efficient!

All pretrained models can be found on [Google Drive](https://drive.google.com/drive/u/1/folders/1SNKb2vJ7laHo0o40n0-OOUjc0kL6b7Yw).

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

Below, we run experiments using ResNet110, varying expansion used for all conv-shift-conv layers in the neural network. Here, we consider 3x3 filters. `a` is the number of filters in the first set of 1x1 convolutional filters. `c` is the number of channels in our input. For comparison, our retrained ResNet110 has accuracy 91.14% and 1.73m parameters.

| Expansion | Acc | Parameters (millions) | Reduction |
|-----------|-----|-----------------------|-----------|
| 1c | 90.34% | 0.20 | 8.5 |
| 2c | 91.56% | 0.40 | 4.4 |
| 3c | |  0.59 | 2.9 |
| 4c | 91.50% |  0.79 | 2.2 |
| 5c | |  0.98 | 1.8 |
| 6c | |  1.18 | 1.5 |
| 7c | |  1.37 | 1.3 |
| 8c | |  1.57 | 1.1 |
| 9c | |  1.76 | 0.98 |
