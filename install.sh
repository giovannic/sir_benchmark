#!/bin/bash

# Install the SIR benchmark
# Requires:
#  * R >= 4.0.0, Python >= 3, make, hyperfine
# Installs:
#  * remotes
#  * individual
#  * sir_cxx R/CPP sir model
#  * mesa
#  * jax
# Builds:
#  * cxx sir model

# install R dependencies
R -e "install.packages('remotes')"
R -e "remotes::install_github('mrc-ide/individual')"
R -e "remotes::install_local('./R/sir_cxx')"

# install Python dependencies
#TODO: GPU version
#pip install mesa "jax[cuda11_pip]"
pip install mesa jax jaxlib jaxtyping

# build CXX model
g++ cxx/sir.cpp -o bin/sir
