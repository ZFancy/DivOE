## Required Packages

The following packages are required to be installed:

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/)
- [Scipy](https://github.com/scipy/scipy)
- [Numpy](http://www.numpy.org/)
- [Sklearn](https://scikit-learn.org/stable/)

All of our experiments are conducted on NVIDIA GeForce RTX451
3090 GPUs with Python 3.7, PyTorch 1.12 and Torchvision 0.13

## Pretrained Models

For CIFAR-10/CIFAR-100, pretrained WRN models are provided in folder

```
./CIFAR/snapshots/
```

For ImageNet, we used the pre-trained ResNet-50 provided by Pytorch.

## Datasets

Please download the datasets in folder

```
./data/
```

### 1. CIFAR-10/100 as ID dataset

#### Auxiliary OOD Dataset

- [tiny-ImageNet-200](https://github.com/chihhuiho/CLAE/blob/main/datasets/download_tinyImagenet.sh)

#### Test OOD Datasets 

- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

- [Places365](http://places2.csail.mit.edu/download.html)

- [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
 
- [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)

### 2. ImageNet as ID dataset
#### Auxiliary OOD Dataset
We employ the
[ImageNet-21K-P dataset](https://arxiv.org/abs/2104.10972) as the auxiliary OOD dataset, which makes invalid classes cleansing and image
resizing compared with the original ImageNet-21K

#### Test OOD Datasets
We employed [iNaturalist](https://arxiv.org/abs/1707.06642), [SUN](https://arxiv.org/abs/1504.06755), [Places365](https://arxiv.org/abs/1610.02055), and [Texture](https://arxiv.org/abs/1311.3618), following the same experiment settings as [MOS](https://arxiv.org/abs/2105.01879).
To download these four test OOD datasets, one could follow the instructions in the [code repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) of MOS.

## Fine-tuning and Testing

run MSP score training and testing for cifar10 WRN
```train
bash run.sh oe_tune 0 
```

run MSP score training and testing for cifar100 WRN
```train
bash run.sh oe_tune 1
```

run MSP score training and testing for ImageNet ResNet-50
```train
bash run.sh oe_tune 2
```

run DivOE with MSP score for extrapolation on cifar10 WRN with the following hyperparameters
$r=0.5,\epsilon=0.05, relative\,step\,size=0.25,k=5$
```train
bash run.sh MSP_DivOE 0 0.5 0.05 0.25 5  
```

run DivOE with MSP score for extrapolation on cifar100 WRN with the following hyperparameters
$r=0.5,\epsilon=0.05, relative\,step\,size=0.25,k=5$
```train
bash run.sh MSP_DivOE 1 0.5 0.05 0.25 5 
```

run DivOE with MSP score for extrapolation on ImageNet ResNet-50 with the following hyperparameters
$r=0.5,\epsilon=0.05, relative\,step\,size=0.25,k=5$
```train
bash run.sh MSP_DivOE 2 0.5 0.05 0.25 5 
```


## Results

Our method achieves the following average performance on all test OOD datasets:

|     | CIFAR-10 | CIFAR-10 | CIFAR-100 | CIFAR-100 | ImageNet | ImageNet |
|:---:|:--------:|:--------:|:---------:|:---------:|:---------:|:---------:|
|     |   FPR95  |   AUROC  |   FPR95   |   AUROC   | FPR95   |   AUROC   |
|  OE |   13.76  |   97.53  |   27.67   |   91.89   |  61.94  |  81.58  |
| DivOE |   **11.66**   |   **97.82**  |   **24.80**   |   **92.91**   |    **60.12**   |   **81.96**   | 