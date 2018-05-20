input_bucket=$1
output_bucket=$2
time_series_A_name=$3
time_series_A_dir=$4
time_series_length=$5
window_size=$6
tile_size=$7
job_queue=$8
job_definition=$9

tile_n=$(($tile_size - $window_size + 1))
scrimp_tile_size=2097152
self_join=1
width=$(($time_series_length / $tile_n))
num_jobs=0

for i in `seq 1 $width`;
do
    num_jobs=$(($num_jobs + $i))
done

echo $num_jobs


#Split up the input and write to s3 requires CLI access to s3

cmd="./run_job_preprocess.sh $input_bucket/$time_series_A_dir/$time_series_A_name.zip $input_bucket split_$time_series_A_dir $window_size $tile_size"
echo $cmd
$cmd


#Requires CLI access to batch
#Requires user specified job queue and job definition
#Submit scrimp job, see documentation for instructions on how to set up
#your own job defintion so that it works with this script
X=`aws batch submit-job --job-name "scrimp-$time_series_A_name" \
                     --job-queue $job_queue \
                     --job-definition $job_definition \
                     --retry-strategy "attempts=2" \
                     --parameters input_bucket=$input_bucket,output_bucket=$output_bucket,output_dir=$time_series_A_name$time_series_A_name,input_dir="split_$time_series_A_dir",num_tiles_wide="$width",tile_width=$tile_size,SCRIMP_Tile_size="$scrimp_tile_size",prefix="segment_",fp64_flag=0 \
                     --array-properties size=$num_jobs \
                     | python -c "import sys, json; print json.load(sys.stdin)['jobId']"`


#wait for job to finish
Y="UNKNOWN"
while [ "$Y" != "SUCCEEDED" ] && [ "$Y" != "FAILED" ];
do
    Y=`aws batch describe-jobs --jobs $X | python -c "import sys, json; print json.load(sys.stdin)['status']"`
    sleep 20s
done

if [ "$Y" = "FAILED" ];
then
    echo "AWS Job Failed"
    exit 1
fi

#pull partial results from s3 and combine into single result
python ./run_job_postprocess.py $output_bucket $time_series_A_dir$time_series_A_dir $tile_n $tile_n $time_series_n $self_join

