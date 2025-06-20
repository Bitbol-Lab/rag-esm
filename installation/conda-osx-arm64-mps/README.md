# Installation with conda

## Cloning the repository

Clone the git repository.

```bash
git clone <HTTPS/SSH> rag-esm
cd rag-esm
```

We will refer the absolute path to the root of the repository as `PROJECT_ROOT`.

## Creating the environment

**Installation**

Create the environment with

```bash
# When in the PROJECT_ROOT directory.
conda env create --file installation/conda-osx-arm64-mps/environment.yml
```

Install the project with

```bash
# Activate the environment
conda activate rag-esm
# When in the PROJECT_ROOT directory.
pip install -e .
```

## Running code in the environment

```bash
conda activate rag-esm
```

Run scripts from the `PROJECT_ROOT` directory.