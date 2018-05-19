#!/bin/bash

if [ $# != 5 ];
then
    echo "Usage: <input s3 time series full file path and bucket> <output s3 bucket> <output s3 dir> <SCRIMP window length> <SCRIMP meta tile size>"
    exit 1
fi

input_s3_ts_filepath=$1
output_s3_bucket=$2
output_s3_dir=$3
window_len=$4
tile_size=$5


cmd="aws s3 cp s3://$input_s3_ts_filepath input.zip"
for i in 1 2 3; do $cmd && break || sleep 5; done

if [ ! -f input.zip ];
then
    echo "Could not pull input from s3"
    exit 1
fi 

unzip input.zip -d input
inputfile=`ls input`

rm input.zip

mkdir splitfiles

cmd="python split_ts.py input/$inputfile $tile_size $window_len splitfiles"
echo $cmd

$cmd

rm -rf input

mkdir $output_s3_dir

cd splitfiles

for file in `ls`;
do
    zip ../$output_s3_dir/$file.zip $file
done

cd ..

rm -rf splitfiles

cmd="aws s3 cp --recursive $output_s3_dir s3://$output_s3_bucket/$output_s3_dir/"
for i in 1 2 3; do $cmd && break || sleep 5; done

rm -rf $output_s3_dir


