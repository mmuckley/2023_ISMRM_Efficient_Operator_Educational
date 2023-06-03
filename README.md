# Efficient Operators

This repository contains code for the Efficien Forward (And Backward!) Models
for Image Reconstruction, given as an educational talk at the 2023 ISMRM Annual
Meeting.

At a high level the repository contains code for reconstructing GRASP data
using modern PyTorch. The repository uses the
[GRASP data](https://cai2r.net/resources/grasp-matlab-code/), which the user is
expected to download separately.

## Installation

These installation instructions require the use of Anaconda. First,
go to the [miniconda site](https://docs.conda.io/en/latest/miniconda.html) and
install miniconda. After that, you should be able to run the commands below.

First, create a new environment to use for the examples:
```bash
conda create -n efficient-operator python=3.10
```

Activate your new environemnt:
```bash
conda activate efficient-operator
```

It's recommended to install the `anaconda` package, which has a wide suite of
basic tools:
```bash
conda install anaconda
```

You will want to separately update ffmpeg so that the videos display
correctly in the notebook:
```bash
conda update ffmpeg
```

After that you can
[install PyTorch using the conda instructions](https://pytorch.org/get-started/locally/).
Once you have PyTorch, you can install `torchkbnufft`
```bash
pip install torchkbnufft
```

And finally, install the local efficient operator (`effop`) package by running
the following in the root directory of this repository:
```bash
pip install -e .
```
The `pip install -e .` command is an editable install, so any changes you make
in the local files should be reflected when you use the package.

## Data configuration

You'll first need to download the data from the following link:

https://cai2r.net/resources/grasp-matlab-code/

After you've downloaded the file, add it to `data_loc.yaml` as
```
path/to/liver_data.mat
```
replacing the path with the true path on your system.

## Running the examples

All of the examples from the presentation are in the `notebooks` folder in
order of slide presentation. The final notebook, `5_full_reconstruction.ibynb`
is self-contained if you want to skip to the end. Note that the outputs have
been wiped from this notebook to comply with the original data agreement
against redistribution on the NYU website.

## Contributing

If you find any mistakes or bugs in the code, please raise an [issue](https://github.com/mmuckley/2023_ISMRM_Efficient_Operator_Educational/issues).

If you have any questions, feel free to open a [discussion](https://github.com/mmuckley/2023_ISMRM_Efficient_Operator_Educational/discussions).
