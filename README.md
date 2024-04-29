# Source code for "Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks"

This repository contains code for reproducing the results in "Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks" [(Sainsbury-Dale, Zammit-Mangion, Richards, and Huser, 2023)](https://arxiv.org/abs/2310.02600).

The methodology described in the manuscript has been incorporated into the Julia package [NeuralEstimators.jl](https://github.com/msainsburydale/NeuralEstimators.jl). In particular, see the example given [here](https://msainsburydale.github.io/NeuralEstimators.jl/dev/workflow/examples/#Irregular-spatial-data). The code in this repository is therefore made available primarily for reproducibility purposes, and we encourage readers seeking to implement GNN-based neural Bayes estimators to explore the package and its documentation. Users are also welcome to contact the package maintainer if there are any questions.

## Repository structure

We first briefly describe the repository structure, although an understanding of this structure is not needed for reproducing the results. The repository is organised into folders containing source code (`src`), intermediate objects generated from the source code (`intermediates`), figures (`img`), and controlling shell scripts that execute the source code (`sh`). Each folder is further divided into the following tree structure, where each branch is associated with one component of the manuscript:

```bash
├── GP                  (Section 3.2)
├── Schlather           (Section 3.3)
├── application         (Section 4)
├── supplement          (Supplementary Material)
```

## Instructions

First, download this repository and navigate to its top-level directory within terminal.

### Software dependencies

Before installing the software dependencies, users may wish to setup a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) environment, so that the dependencies of this repository do not affect the user's current installation. To create a conda environment, run the following command in terminal:

```
conda create -n NeuralEstimatorsGNN -c conda-forge julia=1.9.2 r-base nlopt
```

Then activate the conda environment with:

```
conda activate NeuralEstimatorsGNN
```

The above conda environment installs Julia and R automatically. If you do not wish to use a conda environment, please install [Julia](https://julialang.org/downloads/) and [R](https://www.r-project.org/) manually if they are not already on your system. Once Julia and R are setup, install the Julia and R package dependencies (given in `Project.toml` and `Manifest.toml`, and `dependencies.txt`, respectively) by running the following commands from the top-level of the repository:

```
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```
```
Rscript dependencies_install.R
```

Note that Linux systems have several system dependencies that may need to be installed before installing `devtools` (e.g., `fontconfig1`, `harfbuzz`, and `fribidi`). If you run into problems when installing `devtools`, try installing it manually with  `conda` using the command (other R packages can also be installed this way):

```
conda install -c conda-forge r-devtools
```

or by manually installing the system dependencies before installing `devtools` manually in R:

```
sudo apt -y install libfontconfig1-dev libharfbuzz-dev libfribidi-dev
```


### Hardware requirements

The fast construction of neural Bayes estimators requires graphical processing units (GPUs). Hence, although the code in this repository will run without a GPU (i.e., it will run on the CPU), we recommend that the user run this code on a workstation with a GPU. Note that running the "quick" version of the code (see below) is still fast even on the CPU.

### Reproducing the results

The replication script is `sh/all.sh`, invoked using `bash sh/all.sh` from the top level of this repository. For all studies, the replication script will automatically train the neural estimators, generate estimates from both the neural and likelihood-based estimators, and populate the `img` folder with the figures and results of the manuscript.

The nature of our experiments means that the run time for reproducing the results of the manuscript can be substantial (1-2 days in total). When running the replication script, the user will be prompted with an option to quickly establish that the code is working by using a small number of parameter configurations and epochs. Our envisioned workflow is to establish that the code is working with this "quick" option, clear the populated folders by simply entering `bash sh/clear.sh`, and then run the code in full (possibly over the weekend). **NB:** under this "quick" option, very few training samples and epochs are used when training the GNN, and the results produced will therefore not be meaningful and should not be interpreted.  

Note that the replication script is clearly presented and commented; hence, one may easily "comment out" sections to produce a subset of the results. (Comments in `.sh` files are made with `#`.) By default, results for the max-stable models are not produced, since the simulation of data sets with independent replicates can lead to memory problems on personal computers with limited memory resources; results for these models can be produced by un-commenting the appropriate command in `sh/simulationstudies.sh`. Note that the pairwise likelihood for the Brown-Resnick max-stable process is computed using `C` code, which must be compiled by running the command `bash sh/compile.sh`.


#### Minor reproducibility difficulties

When training neural networks on the GPU, there is some unavoidable non-determinism: see [here](https://discourse.julialang.org/t/flux-reproducibility-of-gpu-experiments/62092). This does not significantly affect the "story" of the final results, but there may be some slight differences each time the code is executed.
