#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

#make result file
#touch test_result.csv
touch test_result.txt

for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "=====START====="
    echo "Running test on $i -> $DEST"
    #echo -n $i >> test_result.csv
    echo $i >> test_result.txt

    # ./sobelf $i $DEST
    salloc -n 1 ./sobelf $i $DEST
done

echo "=====DONE=====" >> test_result.txt
