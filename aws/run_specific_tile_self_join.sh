input_bucket=$1
output_bucket=$2
time_series_A_name=$3
time_series_A_dir=$4
time_series_length=$5
window_size=$6
tile_size=$7
row=$8
col=$9
fp64=${10}
job_queue=${11}
job_definition=${12}

scrimp_tile_size=1048576
self_join=1


#Requires CLI access to batch
#Requires user specified job queue and job definition
#Submit scrimp job, see documentation for instructions on how to set up
#your own job defintion so that it works with this script
X=`aws batch submit-job --job-name "scrimp-$time_series_A_name" \
                     --job-queue $job_queue \
                     --job-definition $job_definition \
                     --retry-strategy "attempts=3" \
                     --parameters input_bucket=$input_bucket,output_bucket=$output_bucket,output_dir=$time_series_A_name$time_series_A_name,input_dir="split_$time_series_A_dir",row_idx="$row",col_idx="$col",tile_width=$tile_size,SCRIMP_Tile_size="$scrimp_tile_size",prefix="segment_",fp64_flag=$fp64,window_size=$window_size \
		     --output 'json' \
		     --query 'jobId'`

if [ $? -ne 0 ];
then
    echo "failed to submit aws job"
    exit 1
fi
