#!/bin/bash

COUNTS=(500 1000 2000 4000 6000 8000 12000)

mkdir -p int_species_scaling

for N in "${COUNTS[@]}"
do

    echo
    echo "=========================================="
    echo "INTEGER LM: $N species"
    echo "=========================================="

    OUT="int_species_scaling/species_${N}.lm"
    LOG="int_species_scaling/species_${N}_run.txt"
    GPU="int_species_scaling/species_${N}_gpu.csv"

    rm -f "$OUT" "$LOG" "$GPU"

    nvidia-smi \
      --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits \
      -lms 100 \
      > "$GPU" &

    GPU_PID=$!

    /usr/bin/time -v \
      python lm_int_species_benchmark.py \
      --species "$N" \
      --particles 100000 \
      -o "$OUT" \
      2>&1 | tee "$LOG"

    STATUS=${PIPESTATUS[0]}

    kill "$GPU_PID" 2>/dev/null || true
    wait "$GPU_PID" 2>/dev/null || true

    if [ "$STATUS" -ne 0 ]; then
        echo
        echo "*** FAILED at $N species ***"
        echo "*** Stopping benchmark series. ***"
        break
    fi

done
