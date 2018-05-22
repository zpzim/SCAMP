input_bucket=$1
output_bucket=$2
time_series_A_name=$3
time_series_A_dir=$4
time_series_A_length=$5
time_series_B_name=$6
time_series_B_dir=$7
time_series_B_length=$8
window_size=$9
fp64=${10}
tile_size=${11}
job_queue=${12}
job_definition=${13}

tile_n=$(($tile_size - $window_size + 1))
time_series_A_n=$(($time_series_A_length - $window_size + 1))
scrimp_tile_size=2097152
self_join=0
width=$(($time_series_A_length / $tile_n))
height=$(($time_series_B_length / $tile_n))
num_jobs=$(($width * $height))

echo $num_jobs


#Split up the input and write to s3 requires CLI access to s3

cmd="./run_job_preprocess.sh $input_bucket/$time_series_A_dir/$time_series_A_name.zip $input_bucket split_$time_series_A_dir $window_size $tile_size"
echo $cmd
$cmd

cmd="./run_job_preprocess.sh $input_bucket/$time_series_B_dir/$time_series_B_name.zip $input_bucket split_$time_series_B_dir $window_size $tile_size"
echo $cmd
$cmd

#Requires CLI access to batch
#Requires user specified job queue and job definition
#Submit scrimp job, see documentation for instructions on how to set up
#your own job defintion so that it works with this script
X=`aws batch submit-job --job-name "scrimp-$time_series_A_name$time_series_B_name" \
                     --job-queue $job_queue \
                     --job-definition $job_definition \
                     --retry-strategy "attempts=2" \
                     --parameters input_bucket=$input_bucket,input_A_dir="split_$time_series_A_dir",input_B_dir="split_$time_series_B_dir",output_bucket=$output_bucket,output_dir=$time_series_A_name$time_series_B_name,prefix="segment_",num_tiles_wide="$width",num_tiles_high="$height",window_size=$window_size,SCRIMP_Tile_size="$scrimp_tile_size",fp64_flag=$fp64 \
                     --array-properties size=$num_jobs \
                     | python -c "import sys, json; print json.load(sys.stdin)['jobId']"`

if [ $? -ne 0 ];
then
    echo "failed to submit aws job"
    exit 1
fi


#wait for job to finish
Y="UNKNOWN"
while [ "$Y" != "SUCCEEDED" ] && [ "$Y" != "FAILED" ];
do
    sleep 20s
    Z=`aws batch describe-jobs --jobs $X`
    Y=`echo $Z | python -c "import sys, json; print json.load(sys.stdin)['jobs'][0]['status']"`
    W=`echo $Z | python -c "import sys, json; print json.load(sys.stdin)['jobs'][0]['arrayProperties']['statusSummary']"`
    echo "Current Job State: $Y"
    echo "Jobs summary: $W"
done

if [ "$Y" = "FAILED" ];
then
    echo "AWS Job Failed"
    exit 1
fi

#pull partial results from s3 and combine into single result
echo "job done running merge routine"
cmd="python ./run_job_postprocess.py $output_bucket $time_series_A_name$time_series_B_name $tile_n $tile_n $time_series_A_n $self_join"
echo $cmd
$cmd

echo "Finished!"
