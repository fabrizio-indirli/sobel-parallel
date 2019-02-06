#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "=====START====="
    echo "Running test on $i -> $DEST"

    #./sobelf $i $DEST
    salloc -n 1 ./sobelf $i $DEST
done
