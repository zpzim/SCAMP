#!/bin/bash

echo "Checking for GPUs"
nvidia-smi | grep 'Driver Version' &> /dev/null
if [ $? != 0 ];
then
   echo "Unable to find an nvidia driver on this system. Aborting!"
   exit 1
fi


if [ $# -lt 12 ];
then
   echo "Usage: <s3 bucket> <s3 input A dir> <s3 input B dir> <s3 output bucket> <s3 output dir> <s3 input file prefix> <number of tile columns> <number of tile rows> <SCRIMP window length> <SCRIMP max tile size> <SCRIMP fp64 flag> <SCRIMP path> <Optional: tile index override>"
   exit 1
fi

bucket=$1
ts_A_dir=$2
ts_B_dir=$3
output_bucket=$4
output_dir=$5
prefix=$6
num_tiles_wide=$7
num_tiles_high=$8

window_len=$9
max_tile_size=${10}
fp_64=${11}
executable_path=${12}
full_join=0

tile_num=$AWS_BATCH_JOB_ARRAY_INDEX
if [ $# -gt 12 ];
then
    tile_num=${13}
fi

if [ $fp_64 == "1" ];
then
    fp_64="-d"
else
    fp_64=""
fi

idx_row=$(($tile_num / $num_tiles_wide))
idx_col=$(($tile_num % $num_tiles_wide))

if [ $idx_row -gt $(($num_tiles_high - 1)) ];
then
    echo "Job unnecessary, tile out of problem space."
    exit 0
fi

echo "tile [$idx_row, $idx_col]"

file_A=$prefix$idx_col
file_B=$prefix$idx_row

cmd="aws s3 cp s3://$bucket/$ts_A_dir/$file_A.zip $file_A.zip"
for i in 1 2 3; do $cmd && break || sleep 5; done

if [ ! -f $file_A.zip ];
then
    echo "Unable to pull input s3://$bucker/$file_A.zip from s3"
    exit 1
fi

unzip -d $file_A $file_A.zip
rm $file_A.zip
x_file_A_name=`ls $file_A`

if [ ! -f $file_A/$x_file_A_name ];
then
    echo "Unable to extract input from archive $file_A.zip"
    exit 1
fi

cmd="aws s3 cp s3://$bucket/$ts_A_dir/$file_B.zip $file_B.zip"
for i in 1 2 3; do $cmd && break || sleep 5; done

if [ ! -f $file_B.zip ];
then
    echo "Unable to pull input s3://$bucket/$file_B.zip from s3"
    exit 1
fi
unzip -d $file_B $file_B.zip
rm $file_B.zip
x_file_B_name=`ls $file_B`

if [ ! -f $file_B/$x_file_B_name ];
then
    echo "Unable to extract input from archive $file_A.zip"
    exit 1
fi

echo Running SCRIMP: $executable_path -s $max_tile_size $fp_64 -b "$file_B/$x_file_B_name" $window_len "$file_B/$x_file_B_name" mpA mpiA
$executable_path -s $max_tile_size $fp_64 -b "$file_B/$x_file_B_name" $window_len "$file_B/$x_file_B_name" mpA mpiA
    
rm -rf $file_A $file_B

if [ ! -f mpA ] || [ ! -f mpiA ];
then
    echo "SCRIMP did not produce output files"
    exit 1
fi

result_file=result_"$idx_row"_"$idx_col".zip
zip $result_file mpA mpiA
rm mpA mpiA

if [ ! -f $result_file ];
then 
    echo "Unable to zip output"
    exit 1
fi
  
cmd="aws s3 cp $result_file s3://$output_bucket/$output_dir/$result_file"
for i in 1 2 3 4 5; do $cmd && break || sleep 5; done

rm $result_file

echo "Finished!"
