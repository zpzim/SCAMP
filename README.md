[![Travis Build Status](https://travis-ci.org/zpzim/SCAMP.svg?branch=master)](https://travis-ci.org/zpzim/SCAMP)
[![Docker Build Status](https://img.shields.io/docker/cloud/build/zpzim/scamp)](https://hub.docker.com/repository/docker/zpzim/scamp)
[![RTD Build Status](https://img.shields.io/readthedocs/scamp-docs)](https://scamp-docs.readthedocs.io/en/latest/)

# SCAMP: SCAlable Matrix Profile

## Table of Contents
[Overview](https://github.com/zpzim/SCAMP#overview) \
[Documentation](https://github.com/zpzim/SCAMP#documentation) \
[Performance](https://github.com/zpzim/SCAMP#performance) \
[Python Module](https://github.com/zpzim/SCAMP#python-module) \
[Run Using Docker](https://github.com/zpzim/SCAMP#run-using-docker) \
[Distributed Operation](https://github.com/zpzim/SCAMP#distributed-operation) \
[Reference](https://github.com/zpzim/SCAMP#reference) \

## Overview
This is a GPU/CPU implementation of the SCAMP algorithm. SCAMP takes a time series as input and computes the matrix profile for a particular window size. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
This is a much improved framework over [GPU-STOMP](https://github.com/zpzim/STOMPSelfJoin) which has the following additional features:
 * Tiling for large inputs 
 * Computation in fp32, mixed fp32/fp64, or fp64 (double is recommended for most datasets, single precision will work for some)
 * fp32 version should get good performance on GeForce cards
 * AB joins (you can produce the matrix profile from 2 different time series)
 * Distributable (we use GCP but other cloud platforms can work) with verified scalability to billions of datapoints
 * More types of matrix profiles! See the Docs!
 * Extremely Efficient Implementation
 * Extensible to adding optimized versions of custom join operations.
 * Can compute joins with the CPU (Only enabled for double precision and does not support all-neighbors joins or distance matrix summaries yet)
 * Handles NaN input values. The matrix profile will be computed while excluding any subsequence with a NaN value
 * Python module: Use SCAMP in Python with pyscamp

## Documentation
SCAMP's documentation can be found at [readthedocs](https://scamp-docs.readthedocs.io/en/latest/).

## Performance
SCAMP is extremely fast, especially on Tesla series GPUs. I belive this repository contains the fastest code in existance for computing the matrix profile. If you find a way to improve the speed of SCAMP, or compute matrix profiles any faster than SCAMP does, please let me know, I would be glad to point to your work and incorporate any improvements that can be made to SCAMP.

More details on the performance of SCAMP can be found in the documentation.

## Python module
A source distribution for a python3 module using pybind11 is available on pypi.org to install run:
~~~
# Python 3 only
pip install pyscamp
~~~

then you can use SCAMP in Python as follows:
~~~
import pyscamp as mp # Uses GPU if available and CUDA was available during the build

# Allows checking if pyscamp was built with CUDA and has GPU support
has_gpu_support = mp.gpu_supported()

# Self join
profile, index = mp.scamp(a, sublen)
# AB join using 4 threads
profile, index = mp.scamp(a, b, sublen, threads=4)
~~~

More information and the API documentation for pyscamp is avalaiable on [readthedocs](https://scamp-docs.readthedocs.io/en/latest/)

## Run Using Docker
You can run SCAMP via [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) using the prebuilt [image](https://hub.docker.com/r/zpzim/scamp) on dockerhub.

In order to expose the host GPUs nvidia-docker must be installed correctly. Please follow the directions provided on the nvidia-docker github page. The following example uses docker 19.03 functionality:
~~~
docker pull zpzim/scamp:latest
docker run --gpus all \
   --volume /path/to/host/input/data/directory:/data \
   --volume /path/to/host/output/directory:/output \
   zpzim/scamp:latest /SCAMP/build/SCAMP \
   --window=<window_size> --input_a_file_name=/data/<filename> \
   --output_a_file_name=/output/<mp_filename> \
   --output_a_index_file_name=/output/<mp_index_filename>
~~~

## Distributed Operation
* We have a client/server architecture built using grpc. Tested on [GKE](https://cloud.google.com/kubernetes-engine/) but should be possible to get working on [Amazon EKS](https://aws.amazon.com/eks/) as well. To use distributed functionality, build the client and server executables via:
~~~
git submodule update --init --recursive
mkdir build && cd build
# requires golang and libz
cmake -DBUILD_CLIENT_SERVER=1 ..
make -j8
~~~
* This will produce three executables in build/kubernetes:
    * "SCAMPserver": This is the SCAMP server. It accepts jobs via grpc and handles divying them up among worker clients.
    * "SCAMPclient": Run this on worker nodes, it must be configured with the hostname and port where the SCAMPserver is. This is the workhorse of the computation, it will utilize all gpus or cpus on the host system to compute work handed to it by the server. Each worker node should have only one client executable running at a time. Though not completely necessary, these clients should have high bandwidth to the server for best performance.
    * "SCAMP_distributed": This behaves similarly to the SCAMP executable above, except that it issues jobs to the server via rpc instead of computing them locally. use the --hostname_port="hostname:port" to configure the address of the server. Currently does not support any kind of authentication, so it will need to be run inside any firewalls that would block internet traffic to the compute cluster.
* The server/clients can be set up to run under kubernetes pods using the Dockerfile in this repo.
* The docker image zpzim/scamp will contain the latest version of the code ready to deploy to kubernetes
* kubernetes/config contains a sample script which will create a GKE cluster using preemptible GPUs and autoscaling as well as sample configuration files for the scamp grpc service, client, and server pods. You should edit these scripts/configuration files to suit your application
* You can use this script to run and execute your own SCAMP workload on GKE as follows:
* Note: The configuration below runs SCAMP_distributed on the server, this is not required and is actually not the desired functionality. We would like to be able to run this remotely. While this is currently possible to do it is not reflected in our example.
~~~
cd kubernetes/config && ./create_gke_cluster.sh
# Once cluster is up and running you can copy your desired input to the server
kubectl cp <local SCAMP input file> <SCAMP server container name>:/
# Now you can run SCAMP_distributed on the server and wait for the job to finish
kubectl exec <SCAMP server container name> -c server -- /SCAMP/build/kubernetes/SCAMP_distributed <SCAMP arguments>
# Copy the results back to a local storage location
kubectl cp <SCAMP server container name>:/mp_columns_out .
~~~
* The above example works on GKE but it should be simple to produce an example that works on Amazon EKS as well.
* Limitations:
    * Server currently does not periodlically save state, so if it dies, all jobs are lost. This will eventually be fixed by adding sever checkpointing.
    * Server currently handles all work in memory and does not write intermediate files to disk. For this reason the server requires a lot of memory to operate on a large input. Eventually the server will operate mostly on files on disk rather than keep all intermediate data in memory.
    * All neighbors profiles and distance matrix summaries are not yet supported in distributed workloads.
#### Sharded implementation
* The original distributed implementation used [AWS batch](https://aws.amazon.com/batch/) and shards the time series to Amazon S3. This approach avoids the above limitations of our in-memory SCAMPserver, however our initial implementation was very limited in scope and was not extensible to other types of SCAMP workloads, so it is mostly obsolete. However, we still provide the scripts used for posterity in the aws/ directory. Though these would be strictly for inspiration, as there are AWS account side configurations required for operation that cannot be provided.

## Reference
If you use SCAMP in your work, please reference the following paper:
~~~
Zimmerman, Zachary, et al. "Matrix Profile XIV: Scaling Time Series Motif Discovery with GPUs to Break a Quintillion Pairwise Comparisons a Day and Beyond." Proceedings of the ACM Symposium on Cloud Computing. 2019.
~~~
