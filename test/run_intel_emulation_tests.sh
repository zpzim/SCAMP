#!/bin/bash

SCAMP_EXECUTABLE=$1
SDE_EXECUTABLE=$2

ARCHS=("-snb -ivb -hsw -bdw -slt -slm -glm -glp -tnt -snr -skl -cnl -icl -skx -clx -cpx -icx -knl -knm -tgl -adl -spr") 
PASSED=""
FAILED=""
for ARCH in $ARCHS;
do
  CMD="${SDE_EXECUTABLE} $ARCH -- ${SCAMP_EXECUTABLE} --window=100 --input_a_file_name=../test/SampleInput/randomwalk8K.txt --no_gpu --num_cpu_workers=1 --print_debug_info"
  echo $CMD
  if $CMD ; then
    PASSED="${PASSED} $ARCH"
  else
    FAILED="${FAILED} $ARCH"
  fi
done

if [ $FAILED ]; then
  echo "Detected Failed Architectures: ${FAILED}"
  exit 1
fi
exit 0

