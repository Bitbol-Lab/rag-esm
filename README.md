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

data: contains some example fasta files to test the model capabilities together with the test set used to evaluate the model in the paper (`clusters_test_set.txt`) and an example training set made of few clusters (`example_dataset`) generated using the code in `src/rag_esm/utils/parse_datasets.ipynb`.

## Getting started

To run the code it is necessary to make an environment with the dependencies listed in the `environment.yml` file. When in the project root directory run:
```bash
conda env create --file installation/conda-osx-arm64-mps/environment.yml
```
To train your model you can use the following command:

```bash
python src/rag_esm/train.py
```

To modify the hyperparameters change the `train.yaml` file in the `src/rag_esm/configs` directory.

## Sample sequences from the model

Example script to sample sequences from the model using the denoising process. The input sequences in `example_input_sequences.fasta` are masked in different ways to show different methods to sample from the model (e.g. using `<mask>` tokens or `-` is interchangeable, also one can decide which parts of the sequences should be masked to allow for inpainting/scaffolding). The context sequences in `example_context_sequences.fasta` are used to condition the generation of the novel sequences. The context sequences can be homologous sequences or any other sequences that can provide useful information for the generation (the latter case is not what the model was trained on, but it can be useful in some cases).

``` bash
python src/rag_esm/sample.py
--training_config_path=/path_to_config_file/config.yaml
--checkpoint_dir=/path_to_weights/
--outputs_dir=/path_to_outputs/
--use_fasta_files
--input_path=data/example_fasta_files/example_input_sequences.fasta
--context_path=data/example_fasta_files/example_context_sequences.fasta
--use_fixed_length_masked_input=100 # this overrides the `input_path` argument and uses a fixed length masked input of 100 amino acids
--batch_size=32 # batch size for the sampling
--error_correction=all-masked # error correction strategy (see sample.py for more details)
--start_error_correction=50 # start error correction after 50 iterations
--mlm_probability=1.0 # (this is overriden by `input_path`) masking probability for the diffusion process (1.0 means masking entirely the input sequences)
--top_k=10 # number of top-k candidates to sample from the model
--top_p=0.0 # top-p sampling (0.0 means no top-p sampling)
--min_p=0.0 # minimum probability for the top-p sampling (0.0 means no minimum probability)
--temperature=1.0 # temperature for the sampling (1.0 means no temperature scaling)
--iterations=100 # number of iterations used for the denoising process
--save_intermediate_steps # save intermediate steps of the denoising process
--compute_perplexities # compute perplexities of the generated sequences using ESM650M
```
## Sample novel sequences from the model conditioned on sequences sampled from the test set clusters

Here we provide an example of how to sample novel sequences from the model conditioned on test set clusters. The input sequences are fully masked and their length is randomly sampled from the length distribution of each cluster. The context sequences are sampled from the same cluster and are used to condition the generation of the novel sequences.

``` bash
python src/rag_esm/sample.py
--training_config_path=/path_to_config_file/config.yaml
--checkpoint_dir=/path_to_weights/
--outputs_dir=/path_to_outputs/
--use_fasta_files
--input_path=data/example_fasta_files/input_sequences_test_set.fasta
--context_path=data/example_fasta_files/context_sequences_test_set.fasta
--batch_size=32
--error_correction=all-masked
--start_error_correction=50
--mlm_probability=1.0
--top_k=10
--iterations=100
--save_intermediate_steps
--compute_perplexities
```

## Sample scaffold sequences from the model conditioned on the original sequence

Here we provide an example of how to sample scaffold sequences around specific motifs (as explained in the paper). The input sequences are composed of the motif inserted into a fully masked sequence of random length. The context sequences are the original sequences from which the motif is extracted. One can introduce more variability in the generation by using different homologs as context.

```bash
python src/rag_esm/sample.py
--training_config_path=/path_to_config_file/config.yaml
--checkpoint_dir=/path_to_weights/
--outputs_dir=/path_to_outputs/
--use_fasta_files
--input_path=data/example_fasta_files/input_sequences_scaffolding.fasta
--context_path=data/example_fasta_files/context_sequences_scaffolding.fasta
--batch_size=32
--error_correction=all-masked
--start_error_correction=50
--temperature=0.7
--top_k=10
--iterations=100
--save_intermediate_steps
--compute_perplexities

```

## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Project Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.

Additional LICENSE files may be present in subdirectories of the project.

