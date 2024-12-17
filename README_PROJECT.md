<div align="center">
  <h1>Paper Title</h1>
  <div>
    <a href='https://first.author.com' target='_blank'>First Author<sup>1</sup></a>;
    <a href='https://Second.author.com' target='_blank'>Second Author<sup>2</sup></a>;
    <a href='https://Third.author.com' target='_blank'>Third Author<sup>3</sup></a>;
  </div>
  <sup>1</sup>First Affiliation
  <sup>2</sup>Second Affiliation
  <sup>3</sup>Third Affiliation
  <br>
  <br>
  <div>
    <a href='https://arxiv.org/abs/<arxiv_id>' target='_blank'><img alt="ArXiv" src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href="https://github.com/DavidZhang73/pytorch-lightning-template"><img alt="Template" src="https://img.shields.io/badge/-Pytorch--Lightning--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  </div>
  <br>
</div>

![Teaser](https://placehold.co/600x300@2x.png?text=Teaser)

## Abstract

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur condimentum bibendum nulla in porta. Fusce id eros diam. Aenean ut egestas tortor, at eleifend felis. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Phasellus ut ante sit amet lorem commodo fringilla. Nulla rhoncus tincidunt erat vitae pulvinar. Vestibulum dapibus, mauris sed pharetra pellentesque, risus tellus fringilla erat, sit amet egestas urna risus eget nulla. Vestibulum porta, mauris sed commodo dictum, metus ex tempor sapien, eget viverra odio enim quis felis. Nunc aliquam nisi non nisl eleifend rhoncus. Duis sit amet mollis libero, porta hendrerit nunc. Maecenas ultricies sapien ultricies, tristique metus eu, blandit felis.

## Method

![Method](https://placehold.co/600x300@2x.png?text=Method)

## Prerequisites

### Installation


```bash
# clone project
git clone https://github.com/YOUR_GITHUB_NAME/YOUR_PROJECT_NAME.git

# create conda virtual environment
conda create -n PROJECT_NAME python=<3.10|3.11|3.12>
conda activate <env_name>

# [Optional] install pytorch according to the official guide to support GPU acceleration, etc.
# https://pytorch.org/get-started/locally/

# install requirements
pip install -r requirements.txt
```

### Data Preparation

### Training

```bash
python src/main.py fit --config configs/data/mnist.yaml --config configs/model/simplenet.yaml --trainer.logger.init_args.name exp1
```

### Inference

```bash
python src/main.py predict --config configs/data/mnist.yaml --config configs/model/simplenet.yaml --trainer.logger.init_args.name exp1
```

## Citation
```
@inproceedings{Key,
  title={Your Title},
  author={Your team},
  booktitle={Venue},
  year={Year}
}
```
