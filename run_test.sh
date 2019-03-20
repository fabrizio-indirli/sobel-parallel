#!/bin/bash

make

INPUT_DIR=images/others
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

rm plog.txt

for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST"

    salloc -n 4 mpirun ./sobelf $i $DEST
done
