#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null


for j in {1..20} ; do
    echo "##### ITERATION $j #####\n"
    for i in $INPUT_DIR/*gif ; do
        DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
        echo "Running test on $i -> $DEST"

        #./sobelf $i $DEST
        salloc -n 4 mpirun sobelf $i $DEST
        # mpirun -np 1 ./sobelf $i $DEST
    done
done