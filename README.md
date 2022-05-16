# Introduction

<p align="center">
  <img width="320" height="200" src="/assets/cifar10_mar_scf.png" hspace="30">
  <img width="320" height="200" src="/assets/imagenet32_mar_scf.png" hspace="30">
</p>

This repository is the PyTorch implementation of the paper:

[**Normalizing Flows with Multi-Scale Autoregressive Priors (CVPR 2020)**](https://arxiv.org/abs/2004.03891)

[Apratim Bhattacharyya<sup>\*</sup>](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/apratim-bhattacharyya/), [Shweta Mahajan<sup>\*</sup>](https://www.visinf.tu-darmstadt.de/team_members/smahajan/smahajan.en.jsp), [Mario Fritz](https://scalable.mpi-inf.mpg.de/), [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele/), [Stefan Roth](https://www.visinf.tu-darmstadt.de/team_members/sroth/sroth.en.jsp)

<sup>\*</sup> <em>Authors contributed equally.</em>

# Getting started

This code has been developed under Python 3.5, Pytorch 1.0.0 and CUDA 9.0.


1. Please run `requirements.py` to check if all required packages are installed.
2. The datasets used in this project are the following,
	- MNIST (included in torchvision.datasets)
	- CIFAR10 (included in torchvision.datasets)
	- ImageNet (available [here](http://image-net.org/small/download.php))

# Training

The script `marscf_main.py` is used for training. The important keyword arguments for training are,
- dataset_name : Name of the dataset in {mnist,cifar10,imagenet_32,imagenet_64} (lowercase string).
- data_root : Path to the location of the dataset. Please see `utils.py` for default values.
- coupling : Type of split coupling to use in the  <em>mAR-SCF</em> model. Possible values {affine,mixlogcdf} (lowercase string).
- batch_size : Recommended values are 128 for affine couplings and 64 for mixlogcdf couplings.
- L : Number of levels in the <em>mAR-SCF</em> model (3 for MNIST, CIFAR10 and ImageNet 32x32; 4 for ImageNet 64x64).
- K : Number of couplings per level.
- C : Number of channels per coupling.

Example usage to train a model on CIFAR10 with MixLogCDF couplings,

	python marscf_main.py --dataset_name cifar10 --coupling mixlogcdf --batch_size 64 --K 4 --C 96

Note, the number of GPUs used can be controlled with the flag `CUDA_VISIBLE_DEVICES`, will default to CPU if no cuda devices are available.

# Generation and Validation

Samples and test results in bits/dim can be obtained using `marscf_main.py` and the flag `--from_checkpoint`. Checkpoints should be stored in the `./checkpoints` folder. Generated samples are stored in the `./samples` folder.

Note, that checkpoints (and sample) files follow the following format,

	marscf_<dataset_name>_<coupling>_<K>_<C>.pt	

Checkpoints can be obtained [here](https://download.visinf.informatik.tu-darmstadt.de/data/2020-cvpr-mahajan-mar-scf/ckpts.zip). Please note that the checkpoints available here for CIFAR10 with MixLogCDF couplings have been trained for longer than reported in the paper, which leads to improved results. E.g. the checkpoint with 256 channels and MixLogCDF couplings have been trained for ~1000 epochs leading to a test NLL (bits/dim) of 3.22 bits/dim and FID of 33.6 (vs 3.24 bits/dim and FID of 41.9 at ~400 epoch as reported in the paper for fair comparison to Residual Flows). 

# Acknowledgement

The code for affine couplings is based on the [glow-pytorch](https://github.com/chaiyujin/glow-pytorch) repo, MixLogCDF couplings on the [flowplusplus](https://github.com/chrischute/flowplusplus) repo and Convolutional LSTM on the [pytorch_convolutional_rnn](https://github.com/kamo-naoyuki/pytorch_convolutional_rnn) repo.

# Bibtex

	@inproceedings{marscf20cvpr,
	title = {Normalizing Flows with Multi-scale Autoregressive Priors},
	author = {Apratim Bhattacharyya and Shweta Mahajan and Mario Fritz and Bernt Schiele and Stefan Roth},
	booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
	year = {2020},
	}
