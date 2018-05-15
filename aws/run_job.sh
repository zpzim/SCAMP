#!/bin/bash

bucket=$1
output_bucket=$2

prefix="$3"_
num_jobs=$4

window_len=$5
max_tile_size=$6
fp_64=$7
self_join=$8
executable_path=$9


idx_row=$(($AWS_BATCH_JOB_ARRAY_INDEX / $num_jobs))
idx_col=$(($AWS_BATCH_JOB_ARRAY_INDEX % $num_jobs))

echo "tile [$idx_row, $idx_col]"

file_A=$prefix$idx_col
file_B=$prefix$idx_row

aws s3 cp s3://$bucket/$file_A.zip $file_A.zip
unzip $file_A.zip
rm $file_A.zip

if [$file_A -ne $file_B];
then
    aws s3 cp s3://$bucket/$file_B.zip $file_B.zip
    unzip $file_B.zip
    rm $file_B.zip
fi

$executable_path $window_len $max_tile_size $fp_64 $self_join $file_A $file_B mpA mpiA mpB mpiB

rm $file_A $file_B

result_file="result_$AWS_BATCH_JOB_ARRAY_INDEX.zip"

zip $result_file mpA mpiA mpB mpiB

rm mpA mpiA mpB mpiB

aws s3 cp $result_file s3://$output_bucket/$result_file

rm $result_file
