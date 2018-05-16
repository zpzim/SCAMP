#!/bin/bash

bucket=$1
ts_A_dir=$2
output_bucket=$3
output_dir=$4
prefix="$5"_
num_tiles_wide=$6

window_len=$7
max_tile_size=$8
fp_64=$9
executable_path=$10
full_join=1

tile_num=$AWS_BATCH_JOB_ARRAY_INDEX

i=0
while [ $(($tile_num - $(($num_tiles_wide - $i)))) -gt -1 ];
do
    if [ $i -eq $(($num_tiles_wide - 1)) ];
    then
	exit
    fi
    tile_num=$(($tile_num - $(($num_tiles_wide - $i))))
    i=$(($i + 1))
done

idx_row=$i
idx_col=$(($tile_num + $i))

echo "tile [$idx_row, $idx_col]"

file_A=$prefix$idx_col
file_B=$prefix$idx_row

aws s3 cp s3://$bucket/$file_A.zip $file_A.zip

unzip -d $file_A $file_A.zip
rm $file_A.zip

x_file_A_name=`ls $file_A`
x_file_B_name=$x_file_A_name
if [ $file_A != $file_B ];
then
    aws s3 cp s3://$bucket/$file_B.zip $file_B.zip
    unzip -d $file_B $file_B.zip
    rm $file_B.zip
    x_file_B_name=`ls $file_B`
fi

$executable_path $window_len $max_tile_size $fp_64 $full_join "$file_A/$x_file_A_name" "$file_B/$x_file_B_name" mpA mpiA mpB mpiB

rm -rf $file_A $file_B

result_file=result_"$idx_row"_"$idx_col".zip

zip $result_file mpA mpiA mpB mpiB

rm mpA mpiA mpB mpiB

aws s3 cp $result_file s3://$output_bucket/$output_dir/$result_file

rm $result_file
