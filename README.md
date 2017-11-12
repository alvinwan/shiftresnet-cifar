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

## Experiments

Below, we run experiments on the following:

1. Varying expansion used for all conv-shift-conv layers in the neural network. Here, we replace 3x3 filters.
2. Varying number of output channels for a 3x3 convolution filter, matching the reduction in parameters that shift provides.

`a` is the number of filters in the first set of 1x1 convolutional filters. `c` is the number of channels in our input.=

All CIFAR-10 pretrained models can be found on [Google Drive](https://drive.google.com/drive/u/1/folders/1rD_b5epthHIDqYSERuwx4gVGcpMcomy7).

### CIFAR-10 Accuracy

| Model | `a` | Acc | Acc (res) | Params* | Reduction** | `r`*** |
|-------|-----|-----|-----------|---------|-------------|--------|
| ResNet20 | c | 86.66% | 85.84% | 0.03 | 7.8 (7.6) | 12 |
| ResNet20 | 3c | 90.08% | 88.33% | 0.10 | 2.9 | 3.3 |
| ResNet20 | 6c | 90.59% | 90.09% | 0.19 | 1.5 | 1.6 |
| ResNet20 | 9c | 91.69% | 91.35% | 0.28 | .98 (1) | 1 |
| ResNet20 | original | - | 91.35% | 0.27 | 1.0 | - |
| ResNet56 | c | 89.71% | 87.46% | 0.10 | 8.4 (8.2) | 16 |
| ResNet56 | 3c | 92.11% | 89.40% | 0.29 | 2.9 | 3.3 |
| ResNet56 | 6c | 92.69% | 89.89% | 0.58 | 1.5 | 1.6 |
| ResNet56 | 9c | 92.74% | 92.01% | 0.87 | 0.98 (0.95) | 0.98 |
| ResNet56 | original | - | 92.01% | 0.86 | 1.0 | - |
| ResNet110 | c | 90.34% | 76.82% | 0.20 | 8.5 (8.2) | 15 |
| ResNet110 | 3c | 91.98% | 74.30% | 0.59 | 2.9 | 3.3 |
| ResNet110 | 6c | 93.17% | 79.02% | 1.18 | 1.5 | 1.6 |
| ResNet110 | 9c | 92.76% | 92.46% | 1.76 | 0.98 (0.95) | 0.98 |
| ResNet110 | original | - | 92.46% | 1.73 | 1.0 | - |

`*` parameters are in the millions

`**` The number in parantheses is the reduction in parameters we used for ResNet, if we could not obtain the exact reduction in parameters used for shift.

`***` Value that you pass to `main.py` for the `--reduction` flag, to reproduce the provided accuracies. This represents the amount to reduce each resnet block's number of "internal convolutional channels" by. In constrast, the column to the left of it is the total neural network's reduction in parameters.

<!--| ResNet110 | 2c | 91.84% | 0.40 | 4.4 |
| ResNet110 | 4c | 91.93% |  0.79 | 2.2 |
| ResNet110 | 5c | 91.77% |  0.98 | 1.8 |
| ResNet110 | 7c | 92.23% |  1.37 | 1.3 |-->

### CIFAR-100 Accuracy

Accuracies below are all Top 1. All CIFAR-100 pretrained models can be found on [Google Drive](https://drive.google.com/drive/u/1/folders/1unOPMsQDagcDa8gI5kFvQ0VH84N7h1V2).

| Model | `a` | Acc | Acc (res) | Params* | Reduction** | `r`*** |
|-------|-----|-----|-----------|---------|-------------|--------|
| ResNet20 | c | ? | ? | 0.03 | 7.8 (7.6) | 12 |
| ResNet20 | 3c | ? | ? | 0.10 | 2.9 | 3.3 |
| ResNet20 | 6c | ? | ? | 0.19 | 1.5 | 1.6 |
| ResNet20 | 9c | ? | ? | 0.28 | .98 (1) | 1 |
| ResNet20 | original | - | 66.25% | 0.27 | 1.0 | - |
| ResNet56 | c | 65.21% | ? | 0.10 | 8.4 (8.2) | 16 |
| ResNet56 | 3c | ? | ? | 0.29 | 2.9 | 3.3 |
| ResNet56 | 6c | ? | ? | 0.58 | 1.5 | 1.6 |
| ResNet56 | 9c | ? | ? | 0.87 | 0.98 (0.95) | 0.98 |
| ResNet56 | original | - | 69.27% | 0.86 | 1.0 | - |
| ResNet110 | c | 67.84% | (a20) | 0.20 | 8.5 (8.2) | 15 |
| ResNet110 | 3c | 71.83% | ? | 0.59 | 2.9 | 3.3 |
| ResNet110 | 6c | (a20) | ? | 1.18 | 1.5 | 1.6 |
| ResNet110 | 9c | (a19) | ? | 1.76 | 0.98 (0.95) | 0.98 |
| ResNet110 | original | - | 72.11% | 1.73 | 1.0 | - |
