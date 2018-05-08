#!/bin/bash

EXECUTABLE=../src/SCRIMP-GPU

WINDOWSZ=(100 500 1000 4000)
TILE_SZ=(1048576 8000 4000 2000 1000 500)
INPUT_FILES=(SampleInput/randomlist8K.txt SampleInput/randomwalk512K.txt SampleInput/randomwalk1M.txt)
COMPARE_MPI_FILES=(SampleOutput/mpi8K SampleOutput/walkmpi512K SampleOutput/walkmp1M)
COMPARE_MP_FILES=(SampleOutput/mp8K SampleOutput/walkmp512K SampleOutput/walkmp1M)
NUM_TESTS=3

for k in `seq 0 $NUM_TESTS`;
do
    INPUT_FILE=${INPUT_FILES[$k]}
    COMPARE_MPI=${COMPARE_MPI_FILES[$k]}
    COMPARE_MP=${COMPARE_MP_FILES[$k]}
    for j in $WINDOWSZ;
    do    
        for i in $TILE_SZ;
        do
            `$EXECUTABLE $j $i $INPUT_FILE mp mpi`
            X=`diff -U 0 mpi $COMPARE_MPI | grep ^@ | wc -l`
            echo "$X flips"
            `python difference.py mp $COMPARE_MPI out`
        done
    done
done
       
rm mp mpi out


