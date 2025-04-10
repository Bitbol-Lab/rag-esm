# Installation with conda

## Cloning the repository

Clone the git repository.

```bash
git clone <HTTPS/SSH> rag-esm
cd rag-esm
```

We will refer the absolute path to the root of the repository as `PROJECT_ROOT`.

## Creating the environment

**Prerequisites**

- `brew`: [Homebrew](https://brew.sh/).
- `mamba` (or equivalently `conda`)

**Installation**

System dependencies:

We list below the important system dependencies that are not available in conda,
but it is hard to list all the system dependencies needed to run the code.
We let you install the missing ones when you encounter errors.

- None.

The conda environment:

Create the environment with

```bash
# When in the PROJECT_ROOT directory.
mamba env create --file installation/conda-osx-arm64-mps/environment.yml
```

Install the project with

```bash
# Activate the environment
mamba activate rag-esm
# When in the PROJECT_ROOT directory.
pip install -e .
```

## Running code in the environment

```bash
mamba activate rag-esm
```

Run scripts from the `PROJECT_ROOT` directory.