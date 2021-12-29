# Setup

## Local

Setting up a machine you can sit in front of or SSH into is easy.

### 1. Check out the repo
```
TODO
```

### 2. Set up the Python environment
We use `conda` for managing Python and CUDA versions, and `pip-tools` for managing Python package dependencies.

We add a `Makefile` for making setup dead-simple.


#### First: Install the Python + CUDA environment using Conda

Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux.

To install `conda`, follow instructions at https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

Note that you will likely need to close and re-open your terminal.
Afterward, you should have ability to run the `conda` command in your terminal.

Run `make conda-update` to create an environment called `fsdl-text-recognizer-2021`, as defined in `environment.yml`.
This environment will provide us with the right Python version as well as the CUDA and CUDNN libraries.

If you edit `environment.yml`, just run `make conda-update` again to get the latest changes.

Next, activate the conda environment.

```sh
conda activate anbnet
```

**IMPORTANT**: every time you work in this directory, make sure to start your session with `conda activate anbnet`.

#### Next: install Python packages

Next, install all necessary Python packages by running `make pip-tools`

Using `pip-tools` lets us do three nice things:

1. Separate out dev from production dependencies (`requirements-dev.in` vs `requirements.in`).
2. Have a lockfile of exact versions for all dependencies (the auto-generated `requirements-dev.txt` and `requirements.txt`).
3. Allow us to easily deploy to targets that may not support the `conda` environment.

If you add, remove, or need to update versions of some requirements, edit the `.in` files, and simply run `make pip-tools` again.

#### Set PYTHONPATH

Last, run `export PYTHONPATH=.` before executing any commands later on, or you will get errors like `ModuleNotFoundError: No module named 'molecule_optimizer'`.

In order to not have to set `PYTHONPATH` in every terminal you open, just add that line as the last line of the `~/.bashrc` file using a text editor of your choice (e.g. `nano ~/.bashrc`)

### Summary

- `environment.yml` specifies python and optionally cuda/cudnn
- `make conda-update` creates/updates the conda env
- `conda activate anbnet` activates the conda env
- `requirements/prod.in` and `requirements/dev.in` specify python package requirements
- `make pip-tools` resolves and install all Python packages
- add `export PYTHONPATH=.:$PYTHONPATH` to your `~/.bashrc` and `source ~/.bashrc`