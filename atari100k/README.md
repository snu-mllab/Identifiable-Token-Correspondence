# Identifiable Token Correspondence for World Models

## Setup

* Use Python 3.10.
* Install [PyTorch](https://pytorch.org/get-started/locally/) (torch and torchvision). Code developed with several versions of Pytorch, with the latest being torch==2.4.1, but should work with other recent versions.
* Install the necessary packages from `requirements.txt`:
```
pip install -r requirements.txt
```
* Warning: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

## Configuration

Configuration files are stored in the `config/benchmark` directory. You can create your own configuration file or pass additional configs as command line arguments directly to the scripts below.

## Policy training

A training run can be launched with

```
python train.py benchmark={config_name} env.train.id={atari_game_name} [{additional_configs} ...]
```

Example:
```
python src/main.py benchmark=atari_itc env.train.id=AsterixNoFrameskip-v4 common.seed=1
```

### Configs
* ITC: `config/benchmark/atari_itc.yaml`
* Baseline (reproduced): `config/benchmark/atari_baseline.yaml`

## Credits

This repository is based on https://github.com/leor-c/simulus

Cohen, L., Wang, K., Kang, B., Gadot, U., and Mannor, S. Uncovering untapped potential in sample-efficient world model agents. arXiv preprint arXiv:2502.11537, 2025.
