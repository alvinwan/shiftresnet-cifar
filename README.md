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

Below, we run experiments using ResNet110, varying expansion used for all conv-shift-conv layers in the neural network. Here, we consider 3x3 filters. `a` is the number of filters in the first set of 1x1 convolutional filters. `c` is the number of channels in our input. For comparison, our retrained ResNet110 has accuracy 91.14%.

| Expansion | a | Acc | Reduction |
|-----------|---|-----|-----------|
| 1 | c | 87.91% | |
| 2 | 2c | 89.02% | |
| 3 | 3c | 89.75% | |
| 4 | 4c | 89.49% | |
| 5 | 5c | 90.37% | |
| 6 | 6c | 91.01% | |
| 7 | 7c | 90.14% | |
| 8 | 8c | 90.96% | |
| 9 | 9c | 90.80% | |
