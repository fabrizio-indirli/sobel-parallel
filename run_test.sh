#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

for j in {1..5}; do
for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST"

    OMP_NUM_THREADS=8 salloc -n 1 -N 1 mpirun ./sobelf $i $DEST # Hybrid
    # ./sobelf $i $DEST
done
done