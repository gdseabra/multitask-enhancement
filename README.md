<div align="center">

# Multitask Enhancement

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is a multitask network latent fingerprint enhancement, orientation field estimation and implicit segmentation.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10.17
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## How to run inference (predictions)

To run the model for inference only, you'll need to:

1. Update the configuration files in the `configs/` directory:

   a. In `configs/predict.yaml`:
   ```yaml
   ckpt_path: path/to/your/model/checkpoint.ckpt  # Path to your trained model checkpoint
   ```

   b. In `configs/data/enhancer_predict.yaml`:
   ```yaml
   data_dir: /path/to/your/data/directory/  # Directory containing the images
   data_list: /path/to/your/data/list.txt   # Text file with list of images to process
   img_subdir: /orig/                       # Subdirectory containing the images
   batch_size: 16                           # Adjust based on your GPU memory
   num_workers: 16                          # Adjust based on your CPU cores
   ```

   c. In `configs/model/enhancer.yaml`:
   ```yaml
   output_path: /path/to/output/directory/  # Where to save the enhanced images
   ```

2. Run the prediction script:

```bash
# Run on CPU
python src/predict.py trainer=cpu

# Run on GPU (recommended)
python src/predict.py trainer=gpu
```

The enhanced images will be saved in the directory specified in the `output_path` configuration.
