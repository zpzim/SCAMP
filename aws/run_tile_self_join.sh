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
   echo "Usage: <s3 bucket> <s3 input A dir> <s3 output bucket> <s3 output dir> <s3 input file prefix> <tile row> <tile col> <tile width> <SCAMP window length> <SCAMP max tile size> <SCAMP fp64 flag> <SCAMP path>"
   exit 1
fi

bucket=$1
ts_A_dir=$2
output_bucket=$3
output_dir=$4
prefix=$5
idx_row=$6
idx_col=$7
tile_width=$8
window_len=$9
max_tile_size=${10}
fp_64=${11}
executable_path=${12}

if [ $fp_64 == "1" ];
then
    fp_64="-d"
else
    fp_64=""
fi

g_start_row=$(($idx_row * ($tile_width - $window_len + 1)))
g_start_col=$(($idx_col * ($tile_width - $window_len + 1)))

echo "tile [$idx_row, $idx_col]"
echo "start [$g_start_row, $g_start_col]"

file_A=$prefix$idx_col
file_B=$prefix$idx_row

cmd="aws s3 cp s3://$bucket/$ts_A_dir/$file_A.zip $file_A.zip"
for i in 1 2 3; do $cmd && break || sleep 5; done

if [ ! -f $file_A.zip ];
then
    echo "Unable to pull input s3://$bucket/$file_A.zip from s3"
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
    echo Running SCAMP: $executable_path -s $max_tile_size $fp_64 -f B -b "$file_B/$x_file_B_name" -r $g_start_row -c $g_start_col $window_len "$file_A/$x_file_A_name" mpA mpiA
    $executable_path -s $max_tile_size $fp_64 -f B -b "$file_B/$x_file_B_name" -r $g_start_row -c $g_start_col $window_len "$file_A/$x_file_A_name" mpA mpiA
    rm -rf $file_A $file_B

    if [ ! -f mpA ] || [ ! -f B_mp ] || [ ! -f mpiA ] || [ ! -f B_mpi ];
    then
        echo "SCAMP did not produce output files"
        exit 1
    fi
    result_file=result_"$idx_row"_"$idx_col"
    mkdir $result_file
    mv mpA $result_file
    mv mpiA $result_file
    mv B_mp $result_file
    mv B_mpi $result_file
    tar cvf $result_file.tar $result_file
    pxz -D 32 -T 32 -0 -cv $result_file.tar > $result_file.tar.xz
else
    echo Running SCAMP: $executable_path -s $max_tile_size $fp_64 $window_len "$file_A/$x_file_A_name" mpA mpiA
    $executable_path -s $max_tile_size $fp_64 $window_len "$file_A/$x_file_A_name" mpA mpiA
    rm -rf $file_A $file_B

    if [ ! -f mpA ] || [ ! -f mpiA ];
    then
        echo "SCAMP did not produce output files"
        exit 1
    fi
    result_file=result_"$idx_row"_"$idx_col"
    mkdir $result_file
    mv mpA $result_file
    mv mpiA $result_file
    tar cvf $result_file.tar $result_file
    pxz -D 32 -T 32 -0 -cv $result_file.tar > $result_file.tar.xz
fi
        


if [ ! -f $result_file.tar.xz ];
then 
    echo "Unable to zip output"
    exit 1
fi
  
cmd="aws s3 cp $result_file.tar.xz s3://$output_bucket/$output_dir/$result_file.tar.xz"
for i in 1 2 3 4 5; do $cmd && break || sleep 5; done

rm $result_file.tar.xz

echo "Finished!"
