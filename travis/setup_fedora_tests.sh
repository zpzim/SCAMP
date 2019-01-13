#!/bin/sh -xe

 # Run tests in Container
sudo docker run --privileged  -v `pwd`:/SCAMP:rw -it fedora:$1 /bin/bash -c "bash -xe /SCAMP/travis/fedora_build_inside_docker.sh $1"

