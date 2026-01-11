<p align="left">
<img src="assets/logo.png" alt="spredle logo" width="150"/>
</p>

# Splicing PREdiction using Deep LEarning (SPREDLE) 

## Overview

spredle is a Python package for predicting RNA splicing from DNA sequences using deep learning models. It is designed to begin by reproducing the SpliceAI (Jaganathan et al. 2019) model in PyTorch, rather than the original TensorFlow v1 implementation, and then to explore additional architectures such as Transformer and Hyena, with the goal of further improving splicing prediction performance. Furthermore, spredle will be extended to predict tissue-specific splicing by applying to long reads RNA-seq data.

## Installation

- using conda

```
git clone git@github.com:HaniceSun/spredle.git
cd spredle
conda env create -f environment.yml
conda activate spredle
```

# Quick Start

```
spredle download-training-data

spredle preprocess --nt 5000 --flank 40

spredle torch-dataset --input Homo_sapiens.GRCh38.115_seq_nt5000_flank40.txt

spredle torch-dataset --input Homo_sapiens.GRCh38.115_seq_nt5000_flank40_sub100.txt

spredle train --config_file config.yaml --model_name SpliceAI-1 --train_file Homo_sapiens.GRCh38.115_seq_nt5000_flank40_sub100_train.pt --val_file Homo_sapiens.GRCh38.115_seq_nt5000_flank40_sub100_val.pt --lr_lambda "1,1,1,1,1,1,0.5,0.25,0.125,0.0625,0.03125,0.015625"

spredle predict --config_file config.yaml --model_name SpliceAI-1 --epoch 9 --pred_file predict.txt

```

## Author and License

**Author:** Han Sun

**Email:** hansun@stanford.edu

**License:** [MIT License](LICENSE)
