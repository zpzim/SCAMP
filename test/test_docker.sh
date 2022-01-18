#!/bin/bash

# This is a basic test that verifies that the docker container works as
# expected. i.e. it checks that scamp runs in the container and outputs
# something, but does not do any correctness checks of scamp itself.
# Correctness checks are left to the other scamp tests.

IMG_AND_TAG=$1
SUBLEN=$2
FILE_A=$3
EXTRA_ARGS=$4
CURR_DIR=`pwd`


docker run \
   --volume $CURR_DIR:/data \
   --volume $CURR_DIR:/output \
   $IMG_AND_TAG /SCAMP/build/SCAMP \
   --window=$SUBLEN --input_a_file_name=/data/$FILE_A \
   --output_a_file_name=/output/mp_columns_out \
   --output_a_index_file_name=/output/mp_columns_out_index \
   $EXTRA_ARGS

if [ ! -f "mp_columns_out" ]; then
  echo "Test Failed!"
  exit 1
fi

NUM_LINES_INPUT=`cat $FILE_A | wc -l`
EXPECTED_LINES_OUTPUT=$((${NUM_LINES_INPUT}-$SUBLEN+1))
NUM_LINES_OUTPUT=`cat mp_columns_out | wc -l`

if [ $NUM_LINES_OUTPUT -eq $EXPECTED_LINES_OUTPUT ]; then
  echo "Test Passed!"
  exit 0
fi

echo "Test Failed!"
exit 1
 
