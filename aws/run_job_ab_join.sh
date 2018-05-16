#!/bin/bash

bucket=$1
ts_A_dir=$2
ts_B_dir=$3
output_bucket=$4
output_dir=$5
prefix="$6"_
num_tiles_wide=$7
num_tiles_tall=$8

window_len=$9
max_tile_size=$10
fp_64=$11
executable_path=$12
full_join=0

tile_num=$AWS_BATCH_JOB_ARRAY_INDEX


idx_row=$(($tile_num / $num_tiles_wide))
idx_col=$(($tile_num % $num_tiles_wide))

if [ $idx_row -gt $(($num_tiles_tall - 1)) ];
then
    exit
fi

echo "tile [$idx_row, $idx_col]"

file_A=$prefix$idx_col
file_B=$prefix$idx_row

aws s3 cp s3://$bucket/$ts_A_dir/$file_A.zip $file_A.zip

unzip -d $file_A $file_A.zip
rm $file_A.zip

x_file_A_name=`ls $file_A`
aws s3 cp s3://$bucket/$ts_B_dir/$file_B.zip $file_B.zip
unzip -d $file_B $file_B.zip
rm $file_B.zip
x_file_B_name=`ls $file_B`

$executable_path $window_len $max_tile_size $fp_64 $full_join "$file_A/$x_file_A_name" "$file_B/$x_file_B_name" mpA mpiA mpB mpiB

rm -rf $file_A $file_B

result_file=result_"$idx_row"_"$idx_col".zip

zip $result_file mpA mpiA mpB mpiB

rm mpA mpiA mpB mpiB

aws s3 cp $result_file s3://$output_bucket/$output_dir/$result_file

rm $result_file
