#!/bin/bash
## Builds a docker image from this repository and uploads it to your configured aws docker registry (according to your default region)
# Verify that this script is actually doing what you want it to do before running it
# Requires aws cli push/pull access to AWS ECR

(cd src/ && make)
X=`nvidia-docker build . | tail -1 | awk '{print $NF}'`

if [ $? -ne 0 ];
then
    echo "Failed to build docker image"
    exit 1
fi

login_cmd=`aws ecr get-login --no-include-email`


if [ $? -ne 0 ];
then
    echo "Failed to get ecr login"
    exit 1
fi

$login_cmd

if [ $? -ne 0 ];
then
    echo "Failed to login to aws docker registry"
    exit 1
fi

tag=`echo $login_cmd | awk '{print $NF}'`
tag=${tag:8}

docker tag $X $tag/SCAMP-gpu:latest

docker push $tag/SCAMP-gpu:latest

