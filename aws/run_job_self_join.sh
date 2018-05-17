#!/bin/bash

echo "Checking for GPUs"
nvidia-smi

if [ $# -lt 10 ];
then
   echo "Usage: <s3 bucket> <s3 input dir> <s3 output bucket> <s3 output dir> <s3 input file prefix> <number of columns in problem> <SCRIMP window length> <SCRIMP max tile size> <SCRIMP fp64 flag> <SCRIMP path> <Optional: tile index override>"
   exit 1
fi

bucket=$1
ts_A_dir=$2
output_bucket=$3
output_dir=$4
prefix="$5"_
num_tiles_wide=$6

window_len=$7
max_tile_size=$8
fp_64=$9
executable_path=${10}
full_join=1

tile_num=$AWS_BATCH_JOB_ARRAY_INDEX
if [ $# -gt 10 ];
then
    tile_num=${11}
fi

i=0

while [ $(($tile_num - $(($num_tiles_wide - $i)))) -gt -1 ];
do
    if [ $i -eq $(($num_tiles_wide - 1)) ];
    then
	echo "Job was unnecessary: previous jobs were enough"
	exit 0
    fi
    tile_num=$(($tile_num - $(($num_tiles_wide - $i))))
    i=$(($i + 1))
done

idx_row=$i
idx_col=$(($tile_num + $i))

echo "tile [$idx_row, $idx_col]"

file_A=$prefix$idx_col
file_B=$prefix$idx_row

aws s3 cp s3://$bucket/$ts_A_dir/$file_A.zip $file_A.zip

if [ ! -f $file_A.zip ];
then
    echo "Unable to pull input s3://$bucker/$file_A.zip from s3"
    exit 1
fi

unzip -d $file_A $file_A.zip
rm $file_A.zip
x_file_A_name=`ls $file_A`
x_file_B_name=$x_file_A_name

if [ ! -f $file_A/$x_file_A_name ];
then
    echo "Unable to extract input from archive $file_A.zip"
    exit 1
fi

if [ $file_A != $file_B ];
then
    aws s3 cp s3://$bucket/$ts_A_dir/$file_B.zip $file_B.zip
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
    echo Running SCRIMP: $executable_path $window_len $max_tile_size $fp_64 $full_join "$file_A/$x_file_A_name" "$file_B/$x_file_B_name" mpA mpiA mpB mpiB
    $executable_path $window_len $max_tile_size $fp_64 $full_join "$file_A/$x_file_A_name" "$file_B/$x_file_B_name" mpA mpiA mpB mpiB

    rm -rf $file_A $file_B

    if [ ! -f mpA ] || [ ! -f mpB ] || [ ! -f mpiA ] || [ ! -f mpiB ];
    then
        echo "SCRIMP did not produce output files"
        exit 1
    fi
    result_file=result_"$idx_row"_"$idx_col".zip
    zip $result_file mpA mpiA mpB mpiB
    rm mpA mpiA mpB mpiB
else
    full_join=0
    echo Running SCRIMP: $executable_path $window_len $max_tile_size $fp_64 $full_join "$file_A/$x_file_A_name" "$file_A/$x_file_A_name" mpA mpiA
    $executable_path $window_len $max_tile_size $fp_64 $full_join "$file_A/$x_file_A_name" "$file_A/$x_file_A_name" mpA mpiA

    rm -rf $file_A $file_B

    if [ ! -f mpA ] || [ ! -f mpiA ];
    then
        echo "SCRIMP did not produce output files"
        exit 1
    fi
    result_file=result_"$idx_row"_"$idx_col".zip
    zip $result_file mpA mpiA
    rm mpA mpiA
fi
        


if [ ! -f $result_file ];
then 
    echo "Unable to zip output"
    exit 1
fi
  
aws s3 cp $result_file s3://$output_bucket/$output_dir/$result_file

rm $result_file

echo "Finished!"
