# ShiftResNet

Train ResNet with shift operations on CIFAR10, CIFAR100 using PyTorch. This uses the [original resnet CIFAR10 codebase](https://github.com/kuangliu/pytorch-cifar.git) written by Kuang Liu. In this codebase, we replace 3x3 convolutional layers with a conv-shift-conv--a 1x1 convolutional layer, a set of shift operations, and a second 1x1 convolutional layer. The repository includes the following:

- training utility to reproduce results
- efficient implementation of the shift layer from [Peter Jin](https://people.eecs.berkeley.edu/~phj/)
- ResNet and ShiftResNet derivatives on CIFAR10/CIFAR100
- count utility for parameters and FLOPs
- evaluation script for offline evaluation
- links to 60+ pretrained models: [CIFAR10 models](https://drive.google.com/open?id=1aszFPLvEDcJsNRBwz-J5eI5VN6Cyc3pP), [CIFAR100 models](https://drive.google.com/open?id=1zjAfTdMlN_EeWxUFD7VSDOak8S1ebzo9)

Unless otherwise specified, the code was written by and experiments were run by [Alvin Wan](http://alvinwan.com) with help from [Bichen Wu](https://github.com/BichenWuUCB).

## [_Shift:_ A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions](https://arxiv.org/pdf/1711.08141.pdf)
By Bichen Wu, Alvin Wan, Xiangyu Yue, Peter Jin, Sicheng Zhao, Noah Golmant, Amir Gholaminejad, Joseph Gonzalez, Kurt Keutzer

Tradeoffs and further analysis can be found in the paper. If you find this work useful for your research, please consider citing:

    @inproceedings{shift,
        Author = {Bichen Wu and Alvin Wan and Xiangyu Yue and Peter Jin and Sicheng Zhao and Noah Golmant and Amir Gholaminejad and Joseph Gonzalez and Kurt Keutzer},
        Title = {Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions},
        Journal = {arXiv:1711.08141},
        Year = {2017}
    }
    

## Getting Started

1. If you have not already, setup a virtual environment with Python2.7, and activate it.

```
virtualenv shift --python=python2.7
source shift/bin/activate
```

Your prompt should now be prefaced with `(shift)`, as in

```
(shift) [user@server:~]$ 
```

2. Install `pytorch` and `torchvision`. Access [pytorch.org](http://pytorch.org), scroll down to the "Getting Started" section, and select the appropriate OS, package manager, Python, and CUDA build. For example, selecting Linux, pip, Python2.7, and CUDA 8 gives the following, as of the time of this writing

```
pip install pytorch torchvision # upgrade to latest PyTorch 0.4.1 official stable version
```

3. Clone the repository

```
git clone --recursive git@github.com:alvinwan/shiftresnet-cifar.git
```

4. `cd` into the cuda layer repository.
```
cd shiftresnet-cifar/models/shiftnet_cuda_v2
```

5. Follow the [ShiftNet Cuda layer instructions](https://github.com/peterhj/shiftnet_cuda_v2), steps 5 and 6:

```
pip install -r requirements.txt
make
```

6. In dir `shiftresnet-cifar/models/shiftnet_cuda_v2`, create an additional `__init__.py` so that Python2 can use `shiftnet_cuda_v2` as a module.

```
touch __init__.py
```

7. Then, `cd` back into the root of this repository. Create the `checkpoint` directory and download a checkpoint.

```
cd ../..
mkdir checkpoint
```

In this example below, we download the original `ResNet20`, 3x smaller `ShiftResNet20-3`, and 3x smaller `ResNet20`. Download [all CIFAR-100 models](https://drive.google.com/open?id=1zjAfTdMlN_EeWxUFD7VSDOak8S1ebzo9). Save these in a `checkpoint` directory, so that your file structure resembles the following:

```
shiftresnet-cifar/
   |
   |-- eval.py
   |-- checkpoint/
       |-- resnet20_cifar100.t7
       |-- ...
```

8. Run the following. This will get you started, downloading the dataset locally to `./data` accordingly. We begin by just evaluating the original ResNet model on CIFAR100.

```
python eval.py --model=checkpoint/resnet20_cifar100.t7 --dataset=cifar100
```

This default ResNet model should give 66.25%. By default, the script loads and trains on CIFAR10. Use the `--dataset` flag, as above, for CIFAR100.

### ShiftNet Expansion

To control the expansion hyperparameter for ShiftNet, identify a ShiftNet architecture and apply expansion. For example, the following uses ResNet20 with Shift modules of expansion `3c`. We should start by counting parameters and FLOPS (for CIFAR10/CIFAR100):

```
python count.py --arch=shiftresnet20 --expansion=3
```

This should output the following parameter and FLOP count:

```
Parameters: (new) 95642 (original) 272474 (reduction) 2.85
FLOPs: (new) 16581248 (original) 40960640 (reduction) 2.47
```

We can then evaluate the associated ShiftResNet, which we downloaded in the first part of this README. Note the arguments to `main.py` and `count.py` are very similar.

```
python eval.py --model=checkpoint/shiftresnet20_3.0_cifar100.t7 --dataset=cifar100
```

The ShiftResNet model above yields 70.77% on CIFAR-100.

### ResNet Reduction

To reduce ResNet by some factor, in terms of its parameters, specify a reduction either block-wise or net-wise. The former reduces the internal channel representation for each BasicBlock. The latter reduces the input and output channels for all convolution layers by half. First, we can check the reduction in parameter count for the entire network. For example, we specify a block-wise reduction of 3x below:

```
python count.py --arch=resnet20 --reduction=2.8 --reduction-mode=block
```

This should output the following parameter and FLOP count:

```
==> resnet20 with reduction 2.80
Parameters: (new) 96206 (original) 272474 (reduction) 2.83
FLOPs: (new) 14197376 (original) 40960640 (reduction) 2.89
```

We again evaluate the associated neural network, which we downloaded in the first part of this README.

```
python eval.py --model=checkpoint/resnet20_2.8_block_cifar100.t7 --dataset=cifar100
```

This reduced ResNet gives 68.30% accuracy on CIFAR-100, 2.47% less than ShiftResNet despite having several hundred more parameters.

## Experiments

Below, we run experiments on the following:

1. Varying expansion used for all conv-shift-conv layers in the neural network. Here, we replace 3x3 filters.
2. Varying number of output channels for a 3x3 convolution filter, matching the reduction in parameters that shift provides. This is `--reduction-mode=block`, which is *not* the default reduction mode.

`a` is the number of filters in the first set of 1x1 convolutional filters. `c` is the number of channels in our input.

### CIFAR-100 Accuracy

Accuracies below are all Top 1. All CIFAR-100 pretrained models can be found on [Google Drive](https://drive.google.com/drive/u/1/folders/1unOPMsQDagcDa8gI5kFvQ0VH84N7h1V2) (It's worth noticing that this pre-trained model is encoded in the python2 way which may cause problems when the model is loaded in a python3 program.). Below, we compare reductions in parameters for the entire net (`--reduction_mode=net`) and block-wise (`--reduction_mode=block`)

| Model | `e` | SRN Acc* | RN Conv Acc | RN Depth Acc | Params | Reduction (conv) | `r`** | `r`*** |
|-------|-----|----------|-------------|--------------|--------|------------------|-------|--------|
| ResNet20  | 1c | 55.05% | 50.23% | **61.32%** | 0.03 | 7.8 (7.2) | 1.12 | 0.38 |
| ResNet20  | 3c | **65.83%** | 60.72% | 64.51% | 0.10 | 2.9 (2.8) | 0.38 | 0.13 | 
| ResNet20  | 6c | **69.73%** | 65.59% | 65.38% | 0.19 | 1.5 | 0.19 | 0.065 |
| ResNet20  | 9c | **70.77%** | 68.30% | 65.59% | 0.28 | .98 | 0.125 | 0.04 |
| ResNet20  | -- | -- | 66.25% | -- | 0.27 | 1.0 | -- | -- |
| ResNet56  | 1c | 63.20% | 58.70% | **65.30%** | 0.10 | 8.4 (7.6) | 1.12 | 0.38 |
| ResNet56  | 3c | **69.77%** | 66.89% | 66.49% | 0.29 | 2.9 | 0.37 | 0.128 |
| ResNet56  | 6c | **72.33%** | 70.49% | 67.46% | 0.58 | 1.5 | 0.19 | 0.065 |
| ResNet56  | 9c | **73.43%** | 71.57% | 67.75% | 0.87 | 0.98 | 0.124 | 0.04 |
| ResNet56  | -- | -- | 69.27% | -- | 0.86 | 1.0 | -- | -- |
| ResNet110 | 1c | **68.01%** | 65.79% | 65.80% | 0.20 | 8.5 (7.8) | 1.1 | 0.37 |
| ResNet110 | 3c | **72.10%** | 70.22% | 67.22% | 0.59 | 2.9 | 0.37 | 0.125 |
| ResNet110 | 6c | **73.17%** | 72.21% | 68.11% | 1.18 | 1.5 | 0.19 | 0.065 |
| ResNet110 | 9c | **73.71%** | 72.67% | 68.39% | 1.76 | 0.98 | 0.123 | 0.04 |
| ResNet110 | -- | -- | 72.11% | -- | 1.73 | 1.0 | -- | -- |

`*` `SRN` ShiftResNet and `RN` ResNet accuracy using convolutional layers (by reducing the number of channels in the intermediate representation of each ResNet block) and using depth-wise convolutional layers (again reducing number of channels in intermediate representation)

`**` This parameter `r` is used for the `--reduction` flag when replicating results for depth-wise convolutional blocks AND for mobilenet blocks.

`***` This parameter `r` is used for the `--reduction` flag with shuffle blocks.

### CIFAR-10 Accuracy

All CIFAR-10 pretrained models can be found on [Google Drive](https://drive.google.com/open?id=1aszFPLvEDcJsNRBwz-J5eI5VN6Cyc3pP) (Same as above, the encoding is in python2 way which is different from python3's encoding).

| Model | `e` | ShiftResNet Acc | ResNet Acc | Params* | Reduction** |
|-------|-----|-----|-----------|---------|-------------|
| ResNet20 | c | 85.78% | 84.77% | 0.03 | 7.8 (7.2) |
| ResNet20 | 3c | 89.56% | 88.81% | 0.10 | 2.9 (2.8) |
| ResNet20 | 6c | 91.07% | 91.30% | 0.19 | 1.5  |
| ResNet20 | 9c | 91.79 | 91.96% | 0.28 | .98 |
| ResNet20 | original | - | 91.35% | 0.27 | 1.0 |
| ResNet56 | c | 89.69% | 88.32% | 0.10 | 8.4 (7.6) |
| ResNet56 | 3c | 92.48% | 91.20% | 0.29 | 2.9 |
| ResNet56 | 6c | 93.49% | 93.01% | 0.58 | 1.5 |
| ResNet56 | 9c | 93.17% | 93.74% | 0.87 | 0.98 |
| ResNet56 | original | - | 92.01% | 0.86 | 1.0 |
| ResNet110 | c | 90.67% | 89.79% | 0.20 | 8.5 (7.8) |
| ResNet110 | 3c | 92.42% | 93.18% | 0.59 | 2.9 |
| ResNet110 | 6c | 93.03% | 93.40% | 1.18 | 1.5 |
| ResNet110 | 9c | 93.36% | 94.09% | 1.76 | 0.98 (0.95) |
| ResNet110 | original | - | 92.46% | 1.73 | 1.0 |

`*` parameters are in the millions

`**` The number in parantheses is the reduction in parameters we used for ResNet, if we could not obtain the exact reduction in parameters used for shift.

`***` If using `--reduction_mode=block`, pass the `reduction` to `main.py` for the `--reduction` flag, to reproduce the provided accuracies. This represents the amount to reduce each resnet block's number of "internal convolutional channels" by. In constrast, the column to the left of it is the total neural network's reduction in parameters.
