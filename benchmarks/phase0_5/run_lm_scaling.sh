#!/bin/bash

mkdir -p lm_scaling

run_test () {

    NAME=$1
    SPECIES=$2
    REACTIONS=$3

    echo
    echo "=========================================="
    echo "$NAME"
    echo "$SPECIES species / $REACTIONS reactions"
    echo "=========================================="

    OUT="lm_scaling/${NAME}.lm"
    LOG="lm_scaling/${NAME}_run.txt"
    GPU="lm_scaling/${NAME}_gpu.csv"

    rm -f "$OUT" "$LOG" "$GPU"

    nvidia-smi \
      --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits \
      -lms 100 \
      > "$GPU" &

    GPU_PID=$!

    /usr/bin/time -v \
      python lm_species_reaction_benchmark.py \
      --species "$SPECIES" \
      --reactions "$REACTIONS" \
      -o "$OUT" \
      2>&1 | tee "$LOG"

    kill "$GPU_PID" 2>/dev/null || true
    wait "$GPU_PID" 2>/dev/null || true
}


# -----------------------------------------
# Species scaling
# roughly one reaction per species
# -----------------------------------------

run_test species_8    8    8
run_test species_32   32   32
run_test species_64   64   64
run_test species_128  128  128
run_test species_224  224  224


# -----------------------------------------
# Reaction scaling
# hold species fixed at 64
# -----------------------------------------

run_test reactions_256    64  256
run_test reactions_1024   64  1024
run_test reactions_4096   64  4096
