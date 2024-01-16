<div align="center">

# Your Project Name

[![Pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Pytorch Lightning](https://img.shields.io/badge/-Lightning-ffffff?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iNDEiIGhlaWdodD0iNDgiIHZpZXdCb3g9IjAgMCA0MSA0OCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIwLjQ5OTIgMEwwIDEyVjM2TDIwLjUgNDhMNDEgMzZWMTJMMjAuNDk5MiAwWk0xNi45NTAxIDM2LjAwMTZMMTkuMTA4OSAyNi42ODU2TDE0LjI1NDggMjEuODkyTDI0LjA3OTEgMTEuOTk5MkwyMS45MTYzIDIxLjMyOTZMMjYuNzQ0NCAyNi4wOTc2TDE2Ljk1MDEgMzYuMDAxNloiIGZpbGw9IiM3OTJFRTUiLz4KPC9zdmc+Cg==)](https://lightning.ai/docs/pytorch/stable/)
[![Pytorch Lightning Template](https://img.shields.io/badge/-Pytorch--Lightning--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/DavidZhang73/pytorch-lightning-template)
[![Conference](http://img.shields.io/badge/CVPR-2022-4b44ce.svg)]()
[![Conference](http://img.shields.io/badge/ICCV-2022-4b44ce.svg)]()
[![Conference](http://img.shields.io/badge/ECCV-2022-4b44ce.svg)]()
[![Paper](http://img.shields.io/badge/Paper-arxiv.1234.1234-B31B1B.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

</div>

## Description
Official PyTorch implementation of your project name.

## How to run

**Data Preparation**

**Installation**

```bash
# clone project
git clone https://github.com/YOUR_GITHUB_NAME/YOUR_PROJECT_NAME.git

# [Optional] create conda virtual environment
conda create -n <env_name> python=<3.8|3.9|3.10>
conda activate <env_name>

# [Optional] use mamba instead of conda
conda install mamba -n base -c conda-forge

# [Optional] install pytorch according to the official guide to support GPU acceleration, etc.
# https://pytorch.org/get-started/locally/

# install requirements
pip install -r requirements.txt
```

**Train**

```bash
python src/main.py fit --config configs/data/mnist.yaml --config configs/model/simplenet.yaml --trainer.logger.init_args.name exp1
```

**Inference**

```bash
python src/main.py predict --config configs/data/mnist.yaml --config configs/model/simplenet.yaml --trainer.logger.init_args.name exp1
```

## Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
