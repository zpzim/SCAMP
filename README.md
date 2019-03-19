[![Build Status](https://travis-ci.org/zpzim/SCAMP.svg?branch=master)](https://travis-ci.org/zpzim/SCAMP)
# SCAMP: SCAlable Matrix Profile
This is a GPU implementation of the SCAMP algorithm. SCAMP takes a time series as input and computes the matrix profile for a particular window size. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
This is a much improved framework over [GPU-STOMP](https://github.com/zpzim/STOMPSelfJoin) which has the following additional features:
 * Tiling for large inputs 
 * Computation in fp32, mixed fp32/fp64, or fp64 (mixed is recommended for most datasets, but if it doesn't work try double precision)
 * fp32 version should be compatible with GeForce cards
 * AB joins (you can produce the matrix profile from 2 different time series)
 * Distributable (we use AWS but other cloud platforms can work) with verified scalability to billions of datapoints
 * Sum and Frequency Joins: rather than compute the nearest neighbor directly, we can compute the sum or frequency of correlations above a threshold (this better describes the frequency of an event, something not obvious from the matrix profile alone)
 * Extensible to adding optimized versions of custom join operations.
 * Some optimizations for architectures other than Volta

Note: for self-joins on small inputs (~2M or less) the features in this repository are probably overkill. You can use [GPU-STOMP](https://github.com/zpzim/STOMPSelfJoin) with little performance difference.

# Environment
This base project requires:
 * Currently builds under Ubuntu/Fedora Linux using gcc/clang and nvcc (if CUDA is available) with cmake (3.8+ for cuda support), this version is not available directly from all package managers so you may need to install it manually from [here](https://cmake.org/download/)
 * Optional (For GPU computation): At least version 9.0 of the CUDA toolkit available [here](https://developer.nvidia.com/cuda-toolkit).
 * Optional: At least version 6.0 of clang (for clang-tidy and clang-format)
 * Google protobufs (v2) must be installed as SCAMP uses this input to communicate internally, protobuf v2.6.x minimum is required, some package managers do not provide this version yet and you will need to install from source [here](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) 
 * An NVIDIA GPU with CUDA support is also required. You can find a list of CUDA compatible GPUs [here](https://developer.nvidia.com/cuda-gpus)
 * Currently Supports Kepler-Volta, but Turing and beyond will likely work as well, just add the -gencode flag for your specific architecture in CMakeLists.txt 
 * Highly recommend using a Pascal/Volta GPU as they are much better (V100 is ~10x faster than a K80 for SCAMP, V100 is ~2-3x faster than a P100)
~~~~
Ubuntu Required Packages:
   sudo apt-get install protobuf-compiler libprotobuf-dev 
   # Depending on Ubuntu version cmake 3.8 may not be available and you will need to install manually
   sudo apt-get install cmake
   # Install cuda via the link above
Fedora:
   sudo dnf install protobuf-devel cmake3 gcc-c++
   # Install cuda via the link above
CentOS:
  yum install cmake3
  # Install protobufs manually from source using the link above 
  # Install cuda via the link above
~~~~
# Usage
~~~~
git clone https://github.com/zpzim/SCAMP
cd SCAMP
git submodule update --init --recursive
# cmake will look in your $PATH for the cuda/c++ compilers
# If you have problems with cmake, you may need to specify a
# cuda or c++ compiler as shown in the next example
cmake .
make -j4
./SCAMP --window=window_size --input_a_file_name=input_A_file_path
~~~~
This will generate two files: mp_columns_out and mp_columns_out_index, which contain the matrix profile and matrix profile index values respectively. 
~~~~
# If you need to specify a specific compiler or cuda toolkit if you have multiple installed, you can use the following defines
# By default cmake will look for cuda at the /usr/local/cuda symlink on linux
cmake -D CMAKE_CUDA_COMPILER=/path/to/nvcc \
      -D CMAKE_CXX_COMPILER=/path/to/clang/or/gcc .
~~~~
* Selected Optional Arguments:
    * "--input_b_file_name=/path/to/file": allows a second input file which acts as the second time series for an AB join. An AB join compares every subsequence in input A with every subsequence in input B, the length of the matrix profile produced by this operation is always determined by input A, but the matrix profile index's values will reference subsequences in input B. Providing this parameter implies that SCAMP will compute an AB join.

    * "--max_tile_size=[integer tile size]": allows you to specify the max tile size used by the SCAMP tile scheme. By default this is set to 1M, but you can adjust this as desired. Note that a tile size smaller than ~1M will likely fail to saturate the compute resources of newer GPUs
    * One of "--double_precision, --mixed_precision, --single_precision": Changes the precision mode of SCAMP, default is double precision, mixed precision will work on many datasets but not all, single precision will work for some simple datasets, but may prove unreliable for many. See test/SampleInput/earthquake_precision_test.txt for an example of a dataset that fails in mixed/single precision. The single precision mode is about 2x faster than double precision, mixed_precision falls in the middle, but can sometimes be as slow as double precision".
    * "--gpus=\"list of device numbers to use\"": allows you to specify which gpus to use on the machine, by default we try to use all of them. The device numbers must be valid cuda devices on your system. You can chain these to add more gpus. Example: --gpus="0 1" will use gpu 0 and gpu 1 on the system.
* There are more arguments that allow you even greater control over what SCAMP can do. Use --helpfull for a list of possible arguments and their descriptions.
* cmake provides support for clang-tidy (when you build) and clang-format (using build target clang-format) to use these please make sure clang-tidy and clang-format are installed on your system

# AWS operation (This functionality is planned to be heavily refactored into a more user-friendly configuration)
* This framework can be used with [AWS Batch](https://aws.amazon.com/batch) to distribute the computation to a cluster of p2 or p3 instances
* Information forthcoming, but the scripts we used to scale out the algorithm are included in the aws/ directory

# TODOs (Contributors welcome):
* Cleanup codebase, improve integration testing
* Add an optimized CPU code path to this framework, we have optimized code [here](https://github.com/kavj/matrixProfile)
* Add documentation for and improve the general usability of the distributed portion of the framework, ease of use and portability would be great
* Add testing infrastructure and additional test cases.



