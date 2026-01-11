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
spredle preprocess 
spredle torch-dataset

spredle train --model_name SpliceAI  --config_file config.yaml --train_file $train_file --val_file $val_file
spredle train --model_name GPT  --config_file config.yaml --train_file $train_file --val_file $val_file
spredle train --model_name hyena  --config_file config.yaml --train_file $train_file --val_file $val_file

```

## Author and License

**Author:** Han Sun

**Email:** hansun@stanford.edu

**License:** [MIT License](LICENSE)
