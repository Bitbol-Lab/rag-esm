# RAG-ESM

## Overview

This repository contains the code for "RAG-ESM": a method for training a RAG model based on the [ESM2 pre-trained models](https://www.science.org/doi/10.1126/science.ade2574). The model uses cross attention layers to improve the performance of ESM2 models by conditioning the generation of masked protein sequences on embeddings of sequences that are homologous to the masked sequence. The model is trained on the [OpenProteinSet](https://registry.opendata.aws/openfold/) dataset.
The model was also trained on the discrete diffusion task using a variable masking probability, therefore it can be used to generate sequences with different levels of noise (masking) via a denoising process.

## Structure of the repository and Reproduction of the results

You will find the code for training the model in the `src` directory. The code is organized as follows:

src/rag_esm: contains the code for the RAG-ESM model
  - `configs`: contains the configuration files used to train the model
    - `train_8M.yaml`: contains the hyperparameters used to train the RAG-ESM (12M) model while `train.yaml` is a more generic configuration file used to train any ESM model.
    - `setup.yaml`: contains base configuration for hydra and wandb setup.
  - `modules`: contains the architecture modules and the the dataset class used to train the model
    - `dataloaders.py`: contains the dataset class used to train the model and the collate function used to create the batches
    - `model.py`: contains the architecture of the RAG-ESM model
    - `esm_decoder.py`: contains the modified ESM2 modules used in the RAG-ESM model. The main differences with respect to ESM-2 are the addition of the cross attention layers and the usage of [Flash-Attention](https://arxiv.org/abs/2205.14135).
  - `utils`: contains some useful functions used to train the model and parse the dataset, e.g. `metrics.py`, `trainer.py` and `hamming.py`, and a jupyter notebook with the code used to create the training set.
    - `generate.py`: contains useful functions to sample sequences from the model. In particular `denoise` can be used to sample using the diffusion process.
  - `train.py`: contains the training script for the model.
  - `sample.py`: contains the script to sample/generate sequences from the model.

## Getting started

To train your model you can use the following command:

```bash
python src/rag_esm/train.py
```

To modify the hyperparameters change the `train.yaml` file in the `src/rag_esm/configs` directory.


## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Project Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.

Additional LICENSE files may be present in subdirectories of the project.
