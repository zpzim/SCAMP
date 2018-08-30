input_bucket=$1
output_bucket=$2
time_series_A_name=$3
time_series_A_dir=$4
time_series_length=$5
window_size=$6
tile_size=$7
fp64=$8
job_queue=$9
job_definition=${10}

tile_n=$(($tile_size - $window_size + 1))
time_series_n=$(($time_series_length - $window_size + 1))
SCAMP_tile_size=1048576
self_join=1
width=`python -c "from math import ceil; print int(ceil($time_series_length / float($tile_n)))"`
echo $width
num_jobs=0

for i in `seq 1 $width`;
do
    num_jobs=$(($num_jobs + $i))
done

echo $num_jobs

#Requires CLI access to batch
#Requires user specified job queue and job definition
#Submit SCAMP job, see documentation for instructions on how to set up
#your own job defintion so that it works with this script
X=`aws batch submit-job --job-name "SCAMP-$time_series_A_name" \
                     --job-queue $job_queue \
                     --job-definition $job_definition \
                     --retry-strategy "attempts=3" \
                     --parameters input_bucket=$input_bucket,output_bucket=$output_bucket,output_dir=$time_series_A_name$time_series_A_name,input_dir="split_$time_series_A_dir",num_tiles_wide="$width",tile_width=$tile_size,SCAMP_Tile_size="$SCAMP_tile_size",prefix="segment_",fp64_flag=$fp64,window_size=$window_size \
                     --array-properties size=$num_jobs \
		     --output 'json' \
		     --query 'jobId'`

if [ $? -ne 0 ];
then
    echo "failed to submit aws job"
    exit 1
fi

temp="${X%\"}"
X="${temp#\"}"

echo Got Job $X
#wait for job to finish
Y="UNKNOWN"
while [ "$Y" != "SUCCEEDED" ] && [ "$Y" != "FAILED" ];
do
    sleep 20s
    Z=`aws batch describe-jobs --jobs $X --output "json"`
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
cmd="python ./run_job_postprocess.py $output_bucket $time_series_A_name$time_series_A_name $tile_n $tile_n $time_series_n $self_join"
echo $cmd
$cmd

echo "Finished!"
