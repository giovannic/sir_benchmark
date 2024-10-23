#!/bin/bash

# ---------------------------------------------------------------
# Plot different implementations of an SIR model for verification
# ---------------------------------------------------------------
# 
PYTHON="python3"
RS="Rscript"
POP="1000"
MAX_SAMPLES=${2:-"10"}
N_PARAMS=${1:-"3"}
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

# Sample parameters
$PYTHON scripts/sample.py $N_PARAMS $PARAM_PATH $SEED

# Copy file contents MAX_SAMPLES times
x=$(cat /tmp/params.csv)
for i in $(seq 1 $MAX_SAMPLES); do
    echo "$x" >> /tmp/params.csv
done

for i in "${!CMDS[@]}"; do
    echo "Running: ${CMDS[$i]}"
    ${CMDS[$i]}
    $PYTHON scripts/plot.py $N_PARAMS /tmp/test.csv "$i.png"
done