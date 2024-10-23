#!/bin/bash

# ---------------------------------------------------
# Benchmark different implementations of an SIR model
# ---------------------------------------------------
# 
# Each implementation takes a path to a csv of model parameters 
# and produces a csv of model outputs for all of the parameters.
# 
# The model parameters used for this benchmark
# (init_infected, beta and gamma)
# are uniformly sampled within viable ranges
# 
# The number of model parameters and size of the simulated 
# population can be passed in as arguments
# 
# Please run ./install.sh to install dependencies
PYTHON="python3"
RS="Rscript"
POP=${1:-"100,1000,10000"}
MAX_SAMPLES=${2:-"10"}
PARAM_PATH=/tmp/params.csv
SEED=42

R_BASE="${RS} R/sir.R"
R_BITSET="${RS} R/sir_bitset.R"
R_VECTOR="${RS} R/sir_vector.R"
R_INDIVIDUAL="${RS} R/sir_individual.R"
PY_MESA="${PYTHON} python/sir_mesa.py"
PY_JAX="${PYTHON} python/sir_jax.py"
CXX="./bin/sir"

POS_ARGS="{pop} $PARAM_PATH /dev/null"
CMDS=""
CMDS+=" '$R_BASE $POS_ARGS'"
CMDS+=" '$R_BITSET $POS_ARGS'"
CMDS+=" '$R_VECTOR $POS_ARGS'"
CMDS+=" '$R_INDIVIDUAL $POS_ARGS'"
CMDS+=" '$PY_MESA $POS_ARGS'"
CMDS+=" '$PY_JAX $POS_ARGS 0'"
# TODO: add GPU version
# CMDS+=" '$PY_JAX $POS_ARGS 1'"
CMDS+=" '$CXX $POS_ARGS'"

$PYTHON scripts/sample.py $MAX_SAMPLES $PARAM_PATH $SEED

echo hyperfine --warmup 1 -L pop $POP $CMDS
eval hyperfine --warmup 1 -L pop $POP $CMDS