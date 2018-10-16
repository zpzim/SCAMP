#!/bin/bash
FMT=$1
result=0
for file in `ls src`
do
    $FMT -style=file src/$file > temp
    X=`diff src/$file temp`
    if [ "$X" != "" ] ; then
        echo $file did not match the formatting guidelines.
        result=1
        echo $X
    fi
done
exit $result 
