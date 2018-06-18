#!/bin/bash

if [ $# != 5 ];
then
    echo "Usage: <input time series file path> <output s3 bucket> <output s3 dir> <SCRIMP window length> <SCRIMP meta tile size>"
    exit 1
fi

input_ts_filepath=$1
output_s3_bucket=$2
output_s3_dir=$3
window_len=$4
tile_size=$5


inputfile=$1

mkdir -p splitfiles

cmd="python split_ts.py $inputfile $tile_size $window_len splitfiles"
echo $cmd

$cmd

if [ $? != 0 ];
then
    echo "Could not split input file"
    exit 1
fi

mkdir -p $output_s3_dir

if [ $? != 0 ];
then
    echo "Could not make output directory"
    exit 1
fi

cd splitfiles

for file in `ls`;
do
    zip ../$output_s3_dir/$file.zip $file
done

cd ..


cmd="aws s3 cp --recursive $output_s3_dir s3://$output_s3_bucket/$output_s3_dir/"
for i in 1 2 3; do $cmd && break || sleep 5; done

if [ $? != 0 ];
then
    echo "Could not upload split input to s3"
    exit 1
fi

rm -rf splitfiles
rm -rf $output_s3_dir


