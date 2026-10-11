#!/bin/bash

mkdir -p storage_scaling

SPECIES_COUNTS=(1024 3000 6000 12000)

for S in "${SPECIES_COUNTS[@]}"
do

    for MODE in dense sparse
    do

        echo
        echo "======================================"
        echo "$MODE : $S species"
        echo "======================================"

        LOG="storage_scaling/${MODE}_${S}.txt"

        /usr/bin/time -v \
          python lowcopy_storage_benchmark.py \
          --mode "$MODE" \
          --spacing-nm 50 \
          --species "$S" \
          --molecules 250000 \
          --steps 100 \
          2>&1 | tee "$LOG"

    done

done
