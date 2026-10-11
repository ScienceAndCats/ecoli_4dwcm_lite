#!/bin/bash

COUNTS=(1000 10000 50000 100000 250000)

for N in "${COUNTS[@]}"
do
    echo "======================================"
    echo "Running $N particles"
    echo "======================================"

    OUT="particle_scaling/${N}.lm"
    LOG="particle_scaling/${N}_run.txt"
    GPU="particle_scaling/${N}_gpu.csv"

    rm -f "$OUT" "$LOG" "$GPU"

    # GPU logger, sampling every 100 ms
    nvidia-smi \
      --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits \
      -lms 100 > "$GPU" &

    GPU_PID=$!

    /usr/bin/time -v \
      python benchmark_50nm.py \
      -n "$N" \
      -o "$OUT" \
      2>&1 | tee "$LOG"

    kill "$GPU_PID"
    wait "$GPU_PID" 2>/dev/null

    echo
done
