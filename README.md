[![Build Status](https://travis-ci.org/zpzim/SCAMP.svg?branch=master)](https://travis-ci.org/zpzim/SCAMP)
# SCAMP: SCAlable Matrix Profile


## Table of Contents
[Overview](https://github.com/zpzim/SCAMP#overview) \
[Profile Types](https://github.com/zpzim/SCAMP#profile-types) \
[Performance](https://github.com/zpzim/SCAMP#performance) \
[Environment](https://github.com/zpzim/SCAMP#environment) \
[Python Module](https://github.com/zpzim/SCAMP#python-module) \
[Configuration](https://github.com/zpzim/SCAMP#configuration) \
[Usage](https://github.com/zpzim/SCAMP#usage) \
[Run Using Docker](https://github.com/zpzim/SCAMP#run-using-docker) \
[Distributed Operation](https://github.com/zpzim/SCAMP#distributed-operation) \
[Examples](https://github.com/zpzim/SCAMP#examples)

## Overview
This is a GPU/CPU implementation of the SCAMP algorithm. SCAMP takes a time series as input and computes the matrix profile for a particular window size. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
This is a much improved framework over [GPU-STOMP](https://github.com/zpzim/STOMPSelfJoin) which has the following additional features:
 * Tiling for large inputs 
 * Computation in fp32, mixed fp32/fp64, or fp64 (double is recommended for most datasets, single precision will work for some)
 * fp32 version should get good performance on GeForce cards
 * AB joins (you can produce the matrix profile from 2 different time series)
 * Distributable (we use GCP but other cloud platforms can work) with verified scalability to billions of datapoints
 * Sum and Frequency Joins: rather than compute the nearest neighbor directly, we can compute the sum or frequency of correlations above a threshold (this better describes the frequency of an event, something not obvious from the matrix profile alone)
 * All-neighbors Joins: rather than return only the nearest neighbor, we can return all matches above a threshold. This can be used in graph-based analytics and also to create low-res (pooled) distance matrices.
 * Distance matrix summaries: SCAMP can return pooled summary versions of the entire distance matrix.
 * Extensible to adding optimized versions of custom join operations.
 * Can compute joins with the CPU (Only enabled for double precision and does not support all-neighbors joins or distance matrix summaries yet)
 * Handles NaN input values. The matrix profile will be computed while excluding any subsequence with a NaN value

## Profile Types
SCAMP can compute various types of matrix profiles:

* 1NN_INDEX: This is the default profile type and the normal definition of matrix profile, it will produce the nearest neighbor distance/correlation of every subsequence as well as the index of the nearest neighbor
* 1NN: This is a slightly faster version of the default profile type, but it only returns the nearest neighbor distance/correlation not the index of the nearest neighbor
* SUM_THRESH: Rather than finding the nearest neighbor, this profile type will compute the sum of the correlations above the specified threshold (--threshold) for each subsequence. This is like a frequency histogram of correlations.
* ALL_NEIGHBORS: [EXPERIMENTAL GPU ONLY, DISTRIBUTED UNSUPPORTED] This returns the approximate K (--max_matches_per_column) nearest neighbors and their correlations/indexes for each subsequence. A threshold (--threshold) can be used to accelerate the computation by ignoring matches below the threshold. This can also be used to produce distance matrix summaries (signifigantly slower and uses more memory currently) using --reduce_all_neighbors, --reduced_height and --reduced_width. See the example below.

All of the above profiles support AB joins

## Performance
SCAMP is extremely fast, especially on Tesla series GPUs. I belive this repository contains the fastest code in existance for computing the matrix profile. If you find a way to improve the speed of SCAMP, or compute matrix profiles any faster than SCAMP does, please let me know, I would be glad to point to your work and incorporate any improvements that can be made to SCAMP.

### Notes on CPU performance
SCAMP's CPU performance is very good. However, how performant it is depends heavily on the compiler you use. Newer compilers are better, clang v6 or greater tends to work best. Newer versions of GCC can work as well. MSVC tends to be slower. There can be up to a 10x (perhaps more) difference depending on the compiler you use. This is related to how different compilers have varying levels of support for autovectorization.

### Performance Comparisons
The included performance tests showcase SCAMP's performance up to an input size of 16M datapoints; however, as we have shown in our publications SCAMP is scalable to hundreds of millions of datapoints and even billions of datapoints with the right hardware.

![SCAMP GPU Performance](/Readme/SCAMP_Profile_Performance_Comparison.png?raw=true "GPU SCAMP Profiles Performance")
In the figure above we show the runtime in seconds for SCAMP's various profile types (self-join) on 2 P100 GPUs.


![SCAMP KNN Performance](/Readme/KNN.png?raw=true "GPU KNN Profiles Performance with different values of K")
In the figure above we show the runtime in seconds for SCAMP's approximate KNN (--profile_type=ALL_NEIGHBORS) matrix profile, while varying K and the input size on 2 P100 GPUs. You can see that SCAMP maintains good performance relative to the baseline 1NN_INDEX matrix profile up to at least K=20, which should be sufficient for almost all practioners. All measurements were made with random data with the initial threshold set to 0 correlation (close to the worst case for KNN).

![SCAMP vs Others](/Readme/other_methods.png?raw=true "Performance of SCAMP on GPU compared to other methods to compute the matrix profile on GPU")
The above figure illustrates SCAMP's performance versus [STUMPY](https://github.com/TDAmeritrade/stumpy) which is a popular matrix profile implementation. As can be seen above SCAMP on 2x P100 GPUs much faster than STUMPY, when STUMPY is running on 16x V100 GPUs, which are about ~2x more powerful than P100s individually. This is several orders of magnitude of difference in processing power.

## Environment
This base project requires:
 * Currently builds under Windows/Mac/Linux using msvc/gcc/clang and nvcc (if CUDA is available) with cmake (3.8+ for cuda support), this version is not available directly from all package managers so you may need to install it manually, the easist way to do this is with python via "pip install cmake" or you can download it manually from [here](https://cmake.org/download/)
 * Optional, but highly recommended: At least version 9.0 of the CUDA toolkit available [here](https://developer.nvidia.com/cuda-toolkit) and an NVIDIA GPU with CUDA (compute capability 3.0+) support. You can find a list of CUDA compatible GPUs [here](https://developer.nvidia.com/cuda-gpus)
 * Currently Supports Kepler-Volta, but Turing and beyond will likely work as well, just add the -gencode flag for your specific architecture in CMakeLists.txt 
 * Highly recommend using a Pascal/Volta GPU as they are much better (V100 is ~10x faster than a K80 for SCAMP, V100 is ~2-3x faster than a P100)
 * If you are using CPUs, using clang-6.0 or above is highly recomended as gcc may not properly autovectorize the CPU kernels.
~~~~
Ubuntu Required Packages:
   # Depending on Ubuntu version cmake 3.8 may not be available and you will need to install manually
   sudo apt-get install cmake
   # Install cuda via the link above
Fedora:
   sudo dnf install cmake3 gcc-c++
   # Install cuda via the link above
CentOS:
  yum install cmake3
  # Install cuda via the link above
~~~~

## Python module
A source distribution for a python3 module using pybind11 is available on pypi.org to install run:
~~~
# Python 3 only; will also install cmake
pip install pyscamp
~~~

then you can use SCAMP in Python as follows:
~~~
import pyscamp as mp # Uses GPU if available and CUDA was available during the build

# Allows checking if pyscamp was built with CUDA and has GPU support
has_gpu_support = mp.has_gpu_support()

# Self join
profile, index = mp.scamp(a, sublen)
# AB join
profile, index = mp.scamp(a, b, sublen)

# KNN
if has_gpu_support:
  knn = mp.knn(a,sublen, k)
  # KNN with threshold
  knn = mp.knn(a, sublen, k, threshold)
~~~~

This is a new feature and still has some kinks to work out. If you have problems building the module (or getting GPU support to work) please submit an issue on github. I don't have access to all build environments so help in addressing these issues is appreciated.

A few notes on GPU support: you need to have a cuda development environment set up in order to build SCAMP with GPU support. If you install pyscamp and it does not detect CUDA during installation it will install using CPU support only. Cmake must detect your cuda installation, this can be especially tricky when using Windows and MSVC as you need to have the CUDA extensions for visual studio installed.

## Configuration
If you need to specify a specific compiler or cuda toolkit if you have multiple installed, you can use the following defines. By default cmake will look for cuda at the /usr/local/cuda symlink on linux
~~~~~
cmake -D CMAKE_CUDA_COMPILER=/path/to/nvcc \
      -D CMAKE_CXX_COMPILER=/path/to/clang++/or/g++ ..
      -D CMAKE_C_COMPILER=/path/to/clang/or/gcc ..
~~~~~

### Forcing cuda/no cuda
You can force cmake to build without cuda using
~~~~
cmake -D FORCE_NO_CUDA=1 ..
~~~~
For testing with cuda, you can force the build to fail if cuda is not found using
~~~~
cmake -D FORCE_CUDA=1 ..
~~~~


## Usage

### Clone the repository and submodules; make a build directory
~~~~
git clone https://github.com/zpzim/SCAMP
cd SCAMP
git submodule update --init --recursive
mkdir build && cd build
~~~~

### Build SCAMP on Mac/Linux/Windows
~~~~
# cmake will look in your $PATH for the cuda/c++ compilers
# If you have problems with cmake, you may need to specify a
# cuda or c++ compiler as shown above
cmake ..
cmake --build . --config Release
~~~~

### Using SCAMP
~~~~
./SCAMP --window=window_size --input_a_file_name=input_A_file_path [--num_cpu_workers=N (to use CPU threads)]
~~~~

This will generate two files: mp_columns_out and mp_columns_out_index, which contain the matrix profile and matrix profile index values respectively. 

* Selected Optional Arguments:
    * "--input_b_file_name=/path/to/file": allows a second input file which acts as the second time series for an AB join. An AB join compares every subsequence in input A with every subsequence in input B, the length of the matrix profile produced by this operation is always determined by input A, but the matrix profile index's values will reference subsequences in input B. Providing this parameter implies that SCAMP will compute an AB join.

    * "--max_tile_size=[integer tile size]": allows you to specify the max tile size used by the SCAMP tile scheme. By default this is set to 1M, but you can adjust this as desired. Note that a tile size smaller than ~1M will likely fail to saturate the compute resources of newer GPUs
    * One of "--double_precision, --mixed_precision, --single_precision": Changes the precision mode of SCAMP, default is double precision, mixed precision will work on many datasets but not all, single precision will work for some simple datasets, but may prove unreliable for many. See test/SampleInput/earthquake_precision_test.txt for an example of a dataset that fails in mixed/single precision. The single precision mode is about 2x faster than double precision, mixed_precision falls in the middle, but can sometimes be as slow as double precision".
    * "--gpus=\"list of device numbers to use\"": allows you to specify which gpus to use on the machine, by default we try to use all of them. The device numbers must be valid cuda devices on your system. You can chain these to add more gpus. Example: --gpus="0 1" will use gpu 0 and gpu 1 on the system.
    * "--num_cpu_workers": allows you to specify the number of cpu threads to compute with, by default we use none. For now, if you don't have gpus, we recommend setting this to the number of cores on your system for best performance. It is possible to perform hetrogeneous GPU/CPU computation using this flag, but because the CPU code isn't optimized yet, you will likely see no speedup compared to using just GPUs
    * "--reduce_all_neighbors": reduces the output of the ALL_NEIGHBORS profile type to a matrix (a summary of the distance matrix) see Examples
* There are more arguments that allow you even greater control over what SCAMP can do. Use --helpfull for a list of possible arguments and their descriptions.
* cmake provides support for clang-tidy (when you build) and clang-format (using build target clang-format) to use these please make sure clang-tidy and clang-format are installed on your system

## Run Using Docker
Rather than building from scratch you can run SCAMP via [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) using the prebuilt [image](https://hub.docker.com/r/zpzim/scamp) on dockerhub.

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


## Examples

### Distance Matrix Summaries using --reduce_all_neighbors --reduced_height --reduced_width

![Distance Matrix Summary](/Readme/distance_matrix_summary.png?raw=true "Distance Matrix Summary")

You can see that various behavors in the data become apparent through the visualization of the distance matrix.

## References
If you use SCAMP in your work, please reference the following paper:
~~~~
Zimmerman, Zachary, et al. "Matrix Profile XIV: Scaling Time Series Motif Discovery with GPUs to Break a Quintillion Pairwise Comparisons a Day and Beyond." Proceedings of the ACM Symposium on Cloud Computing. 2019.
~~~~
