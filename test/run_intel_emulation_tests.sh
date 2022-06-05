#!/bin/bash

SCAMP_EXECUTABLE=$1
SDE_EXECUTABLE=$2

ARCHS=("-p4p -mrm -pnr -nhm -wsm -snb -ivb -hsw -bdw -slt -slm -glm -glp -tnt -snr -skl -cnl -icl -skx -clx -cpx -icx -knl -knm -tgl -adl -spr") 
PROFILES=("1NN_INDEX 1NN SUM_THRESH MATRIX_SUMMARY")
PASSED=""
FAILED=""
for ARCH in $ARCHS;
do
  for PROFILE in $PROFILES;
  do
    CMD="${SDE_EXECUTABLE} -chip-check-exe-only -chip_check_call_stack $ARCH -- ${SCAMP_EXECUTABLE} --window=100 --input_a_file_name=../test/SampleInput/randomwalk8K.txt --profile_type=$PROFILE --no_gpu --num_cpu_workers=1 --print_debug_info"
    echo $CMD
    eval $CMD
    ret=$?
    if [ $ret -eq 0 ] ; then
      echo "PASS"
      PASSED="${PASSED} $PROFILE/$ARCH"
    else
      echo "FAIL"
      FAILED="${FAILED} $PROFILE/$ARCH"
    fi
  done
done

if [ $FAILED ]; then
  echo "Detected Failed Architectures: ${FAILED}"
  exit 1
fi
exit 0
