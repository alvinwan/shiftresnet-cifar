# ShiftResNet

Train ResNet with shift operations on CIFAR10, CIFAR100 using PyTorch. This uses the [original resnet CIFAR10 codebase](https://github.com/kuangliu/pytorch-cifar.git) written by Kuang Liu. In this codebase, we replace 3x3 convolutional layers with a conv-shift-conv--a 1x1 convolutional layer, a set of shift operations, and a second 1x1 convolutional layer. The repository includes the following:

- utility for training ResNet and ShiftResNet derivatives on CIFAR10/CIFAR100
- count utility for parameters and FLOPs
- evaluation script for offline evaluation
- links to 60+ pretrained models ([CIFAR10 models](https://drive.google.com/drive/u/1/folders/1rD_b5epthHIDqYSERuwx4gVGcpMcomy7), [CIFAR100 models](https://drive.google.com/drive/u/1/folders/1unOPMsQDagcDa8gI5kFvQ0VH84N7h1V2))

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

> **Getting `invalid_device_function`?** Update the architecture code in [`models/shiftnet_cuda_v2/Makefile`](https://github.com/alvinwan/shiftresnet-cifar/blob/master/models/shiftnet_cuda_v2/Makefile#L4), currently configured for a Titan X. e.g., A Tesla K80 is `sm-30`.

3. Run the following. This will get you started, downloading the dataset locally to `./data` accordingly.

```
python main.py
```

By default, the script loads and trains on CIFAR10. Use the `--dataset` flag for CIFAR100.

### ShiftNet Expansion

To control the expansion hyperparameter for ShiftNet, identify a ShiftNet architecture and apply expansion. For example, the following uses ResNet20 with Shift modules of expansion `3c`. We should start by counting parameters and FLOPS (for CIFAR10/CIFAR100):

```
python count.py --arch=shiftresnet20 --expansion=3
```

This should output the following parameter and FLOP count:

```
Parameters: (new) 95642 (original) 272474 (reduction) 2.85
FLOPs: (new) 25886720 (original) 66781184 (reduction) 2.58
```

We can then train the specified ShiftResNet. Note the arguments to `main.py` and `count.py` are very similar.

```
python main.py --arch=shiftresnet20 --expansion=3
```

### ResNet Reduction

To reduce ResNet by some factor, in terms of its parameters, specify a reduction either block-wise or net-wise. The former reduces the internal channel representation for each BasicBlock. The latter reduces the input and output channels for all convolution layers by half. First, we can check the reduction in parameter count for the entire network. For example, we specify a block-wise reduction of 3x below:

```
python count.py --arch=resnet20 --reduction=5 --reduction-mode=block
```

This should output the following parameter and FLOP count:

```
==> resnet20 with reduction 5.00
Parameters: (new) 66264 (original) 272474 (reduction) 4.11
FLOPs: (new) 18776064 (original) 66781184 (reduction) 3.56
```

Once you have a `--reduction` parameter you're happy with, we can run the following to train a reduced ResNet.

```
python main.py --arch=resnet20 --reduction=5 --reduction-mode=block
```

## Experiments

Below, we run experiments on the following:

1. Varying expansion used for all conv-shift-conv layers in the neural network. Here, we replace 3x3 filters.
2. Varying number of output channels for a 3x3 convolution filter, matching the reduction in parameters that shift provides. This is `--reduction-mode=block`, which is *not* the default reduction mode.

`a` is the number of filters in the first set of 1x1 convolutional filters. `c` is the number of channels in our input.=

All CIFAR-10 pretrained models can be found on [Google Drive](https://drive.google.com/drive/u/1/folders/1rD_b5epthHIDqYSERuwx4gVGcpMcomy7).

### CIFAR-10 Accuracy

| Model | `a` | ShiftResNet Acc | ResNet Acc | Params* | Reduction** | `r`*** |
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
| ResNet110 | 9c | 92.79% | 92.46% | 1.76 | 0.98 (0.95) | 0.98 |
| ResNet110 | original | - | 92.46% | 1.73 | 1.0 | - |

`*` parameters are in the millions

`**` The number in parantheses is the reduction in parameters we used for ResNet, if we could not obtain the exact reduction in parameters used for shift.

`***` If using `--reduction_mode=block`, this is the value that you pass to `main.py` for the `--reduction` flag, to reproduce the provided accuracies. This represents the amount to reduce each resnet block's number of "internal convolutional channels" by. In constrast, the column to the left of it is the total neural network's reduction in parameters.

<!--| ResNet110 | 2c | 91.84% | 0.40 | 4.4 |
| ResNet110 | 4c | 91.93% |  0.79 | 2.2 |
| ResNet110 | 5c | 91.77% |  0.98 | 1.8 |
| ResNet110 | 7c | 92.23% |  1.37 | 1.3 |-->

### CIFAR-100 Accuracy

Accuracies below are all Top 1. All CIFAR-100 pretrained models can be found on [Google Drive](https://drive.google.com/drive/u/1/folders/1unOPMsQDagcDa8gI5kFvQ0VH84N7h1V2). Below, we compare reductions in parameters for the entire net (`--reduction_mode=net`) and block-wise (`--reduction_mode=block`)

| Model | `a` | ShiftResNet Acc | ResNet Acc (block)* | ResNet Acc (net) | Params | Reduction | `r` |
|-------|-----|-----|-----------------|--------------|---------|-------------|--------|
| ResNet20 | c | 55.62% | 52.40% | 49.58% | 0.03 | 7.8 (7.6) | 12 |
| ResNet20 | 3c | 62.32% | 60.61% | 58.16% | 0.10 | 2.9 | 3.3 |
| ResNet20 | 6c | 68.64% | 64.27% | 63.22% | 0.19 | 1.5 | 1.6 |
| ResNet20 | 9c | 69.82% | 66.25% | 65.31% | 0.28 | .98 (1) | 1 |
| ResNet20 | original | - | 66.25% | - | 0.27 | 1.0 | - |
| ResNet56 | c | 65.21% | 56.78% | 56.62% | 0.10 | 8.4 (8.2) | 16 |
| ResNet56 | 3c | 69.77% | 62.53% | 64.49% | 0.29 | 2.9 | 3.3 |
| ResNet56 | 6c | 72.13% | 61.99% | 67.45% | 0.58 | 1.5 | 1.6 |
| ResNet56 | 9c | 73.64% | 69.27% | 68.63% | 0.87 | 0.98 (0.95) | 0.98 |
| ResNet56 | original | - | 69.27% | - | 0.86 | 1.0 | - |
| ResNet110 | c | 67.84% | 39.90% | 60.44% | 0.20 | 8.5 (8.2) | 15 |
| ResNet110 | 3c | 71.83% | 40.52% | 66.61% | 0.59 | 2.9 | 3.3 |
| ResNet110 | 6c | 72.56% | 40.23% | 68.87% | 1.18 | 1.5 | 1.6 |
| ResNet110 | 9c | 74.10% | 72.11% | 70.14% | 1.76 | 0.98 (0.95) | 0.98 |
| ResNet110 | original | - | 72.11% | - | 1.73 | 1.0 | - |

`*` ResNet accuracy using block-wise reduction.
