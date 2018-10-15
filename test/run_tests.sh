#!/bin/bash

if [ $# != 1 ];
then
    echo "Usage: run_tests.sh <fp64 flag (0 fp32, 1 fp64)>"
fi


fp64=$1

EXECUTABLE=../SCAMP
ROOT_DIR_INPUT=SampleInput
ROOT_DIR_OUTPUT=SampleOutput
WINDOWSZ=(100)
TILE_SZ=(2000000 1000000 500000 250000 125000 62500 31250 15625 8000 4000)
INPUT_FILES=(randomwalk8K randomwalk16K randomwalk32K randomwalk64K randomwalk1M)
AB_INPUT_FILES=(randomwalk16K randomwalk32K)
NUM_TESTS=$((${#INPUT_FILES[@]} - 1))
NUM_TILE_SZ=$((${#TILE_SZ[@]} - 1))
NUM_AB=$((${#AB_INPUT_FILES[@]} - 1))

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
            if [ $tile_sz -lt $(($count * 2)) ]; then
                echo "Running Test: $EXECUTABLE $j -s $tile_sz $fp64 $INPUT_FILE mp mpi"
                $EXECUTABLE $j -s $tile_sz $fp64 $INPUT_FILE mp mpi > /dev/null
                X=`diff --suppress-common-lines --speed-large-files -y $COMPARE_MPI mpi | grep '^' | wc -l`
                echo "$X matrix profile index differences"
                ./difference.py mp $COMPARE_MP out
            fi
        done
    done
done

for i in `seq 0 $NUM_AB`;
do
    for j in `seq $(($i + 1)) $NUM_AB`;
    do
        INPUT_FILE_A=$ROOT_DIR_INPUT/${AB_INPUT_FILES[$i]}.txt
        INPUT_FILE_B=$ROOT_DIR_INPUT/${AB_INPUT_FILES[$j]}.txt
        for k in $WINDOWSZ;
        do
            COMPARE_MP=$ROOT_DIR_OUTPUT/mp_${AB_INPUT_FILES[$i]}_${AB_INPUT_FILES[$j]}_w$k.txt
            COMPARE_MPI=$ROOT_DIR_OUTPUT/mpi_${AB_INPUT_FILES[$i]}_${AB_INPUT_FILES[$j]}_w$k.txt
            for l in `seq 0 $NUM_TILE_SZ`;
            do
                tile_sz=${TILE_SZ[$l]}
                count=`wc -l $INPUT_FILE_A | awk '{print $1}'`
                if [ $tile_sz -lt $(($count * 2)) ]; then
                    echo "Running Test: $EXECUTABLE -s $tile_sz -b $INPUT_FILE_B $fp64 $k $INPUT_FILE_A mp mpi"
                    $EXECUTABLE -s $tile_sz -b $INPUT_FILE_B $fp64 $k $INPUT_FILE_A mp mpi > /dev/null
                    X=`diff --suppress-common-lines --speed-large-files -y $COMPARE_MPI mpi | grep '^' | wc -l`
                    echo "$X matrix profile index differences"
                    ./difference.py mp $COMPARE_MP out
                fi
            done
        done
        for k in $WINDOWSZ;
        do
            COMPARE_MP=$ROOT_DIR_OUTPUT/mp_${AB_INPUT_FILES[$j]}_${AB_INPUT_FILES[$i]}_w$k.txt
            COMPARE_MPI=$ROOT_DIR_OUTPUT/mpi_${AB_INPUT_FILES[$j]}_${AB_INPUT_FILES[$i]}_w$k.txt
            for l in `seq 0 $NUM_TILE_SZ`;
            do
                tile_sz=${TILE_SZ[$l]}
                count=`wc -l $INPUT_FILE_A | awk '{print $1}'`
                if [ $tile_sz -lt $(($count * 2)) ]; then
                    echo "Running Test: $EXECUTABLE -s $tile_sz -b $INPUT_FILE_A $fp64 $k $INPUT_FILE_B mp mpi"
                    $EXECUTABLE -s $tile_sz -b $INPUT_FILE_A $fp64 $k $INPUT_FILE_B mp mpi > /dev/null
                    X=`diff --suppress-common-lines --speed-large-files -y $COMPARE_MPI mpi | grep '^' | wc -l`
                    echo "$X matrix profile index differences"
                    ./difference.py mp $COMPARE_MP out
                fi
            done
        done
        for k in $WINDOWSZ;
        do
            COMPARE_MP=$ROOT_DIR_OUTPUT/mp_${AB_INPUT_FILES[$j]}_${AB_INPUT_FILES[$i]}_w$k.txt
            COMPARE_MPI=$ROOT_DIR_OUTPUT/mpi_${AB_INPUT_FILES[$j]}_${AB_INPUT_FILES[$i]}_w$k.txt
            COMPARE_MPB=$ROOT_DIR_OUTPUT/mp_${AB_INPUT_FILES[$i]}_${AB_INPUT_FILES[$j]}_w$k.txt
            COMPARE_MPIB=$ROOT_DIR_OUTPUT/mpi_${AB_INPUT_FILES[$i]}_${AB_INPUT_FILES[$j]}_w$k.txt
            for l in `seq 0 $NUM_TILE_SZ`;
            do
                tile_sz=${TILE_SZ[$l]}
                count=`wc -l $INPUT_FILE_A | awk '{print $1}'`
                if [ $tile_sz -lt $(($count * 2)) ]; then
                    echo "Running Test: $EXECUTABLE -s $tile_sz -b $INPUT_FILE_A -f B -r 9000000000 -c 100000 $fp64 $k $INPUT_FILE_B mp mpi"
                    $EXECUTABLE -s $tile_sz -b $INPUT_FILE_A -f B -r 9000000000 -c 100000 $fp64 $k $INPUT_FILE_B mp mpi > /dev/null
                    echo "Checking AB result"
                    X=`diff --suppress-common-lines --speed-large-files -y $COMPARE_MPI mpi | grep '^' | wc -l`
                    echo "$X matrix profile index differences"
                    ./difference.py mp $COMPARE_MP out
                    echo "Checking BA result"
                    X=`diff --suppress-common-lines --speed-large-files -y $COMPARE_MPIB B_mpi | grep '^' | wc -l`
                    echo "$X matrix profile index differences"
                    ./difference.py B_mp $COMPARE_MPB out

                    echo "Running Test: $EXECUTABLE -s $tile_sz -b $INPUT_FILE_B -f B -r 9000000000 -c 100000 $fp64 $k $INPUT_FILE_A mp mpi"
                    $EXECUTABLE -s $tile_sz -b $INPUT_FILE_B -f B -r 9000000000 -c 100000 $fp64 $k $INPUT_FILE_A mp mpi > /dev/null
                    echo "Checking AB result"
                    X=`diff --suppress-common-lines --speed-large-files -y $COMPARE_MPIB mpi | grep '^' | wc -l`
                    echo "$X matrix profile index differences"
                    ./difference.py mp $COMPARE_MPB out
                    echo "Checking BA result"
                    X=`diff --suppress-common-lines --speed-large-files -y $COMPARE_MPI B_mpi | grep '^' | wc -l`
                    echo "$X matrix profile index differences"
                    ./difference.py B_mp $COMPARE_MP out
                fi
            done
        done
    done
done

rm mp mpi out B_mp B_mpi

