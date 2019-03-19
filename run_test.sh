#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

rm plog.txt

for i in {1..5} ; do
    for i in $INPUT_DIR/*gif ; do
        DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
        echo "Running test on $i -> $DEST"

        salloc -n 1 mpirun ./sobelf $i $DEST
    done
done
