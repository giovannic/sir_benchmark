#!/bin/bash

# ---------------------------------------------------------------
# Plot different implementations of an SIR model for verification
# ---------------------------------------------------------------
# 
PYTHON="python3"
RS="Rscript"
POP="1000"
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

POS_ARGS="$POP $PARAM_PATH /tmp/test.csv"

CMDS=()
CMDS+=("$R_BASE $POS_ARGS")
CMDS+=("$R_BITSET $POS_ARGS")
CMDS+=("$R_VECTOR $POS_ARGS")
CMDS+=("$R_INDIVIDUAL $POS_ARGS")
CMDS+=("$PY_MESA $POS_ARGS")
CMDS+=("$PY_JAX $POS_ARGS 0")
# TODO: add GPU version
# CMDS+=("$PY_JAX $POS_ARGS 1")
CMDS+=("$CXX $POS_ARGS")

$PYTHON scripts/sample.py $MAX_SAMPLES $PARAM_PATH $SEED

for i in "${!CMDS[@]}"; do
    echo "Running: ${CMDS[$i]}"
    ${CMDS[$i]}
    head -n 10 /tmp/test.csv
    $PYTHON scripts/plot.py /tmp/test.csv "$i.png"
done