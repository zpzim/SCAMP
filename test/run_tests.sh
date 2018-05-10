#!/bin/bash

EXECUTABLE=../src/SCRIMP-GPU
ROOT_DIR_INPUT=SampleInput
ROOT_DIR_OUTPUT=SampleOutput
WINDOWSZ=(100)
TILE_SZ=(2000000 1000000 500000 250000 125000 62500 31250 15625 8000 4000)
INPUT_FILES=(randomwalk8K randomwalk16K randomwalk32K randomwalk64K randomwalk1M)
NUM_TESTS=4
NUM_TILE_SZ=9

for k in `seq 0 $NUM_TESTS`;
do
    INPUT_FILE=$ROOT_DIR_INPUT/${INPUT_FILES[$k]}.txt
    
    for j in $WINDOWSZ;
    do    
        COMPARE_MPI=$ROOT_DIR_OUTPUT/mpi_${INPUT_FILES[$k]}_w$j.txt
        COMPARE_MP=$ROOT_DIR_OUTPUT/mp_${INPUT_FILES[$k]}_w$j.txt
        for i in `seq 0 $NUM_TILE_SZ`;
        do
            tile_sz=${TILE_SZ[i]}
            count=`wc -l $INPUT_FILE | awk '{print $1}'`
            echo $(($count * 2))
            if [ $tile_sz -lt $(($count * 2)) ]; then
                echo "$EXECUTABLE $j $tile_sz $INPUT_FILE mp mpi 0"
                `$EXECUTABLE $j $tile_sz $INPUT_FILE mp mpi 0`
                X=`diff -U 0 mpi $COMPARE_MPI | grep ^@ | wc -l`
                echo "$X flips"
                echo ./difference.py mp $COMPARE_MP out
                ./difference.py mp $COMPARE_MP out
            fi
        done
    done
done

rm mp mpi out

