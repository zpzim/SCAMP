# GPU-SCRIMP
This is a GPU implementation of the SCRIMP algorithm. SCRIMP takes a time series as input and computes the matrix profile for a particular window size. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
This is a more fleshed out implementation of [GPU-STOMP](https://github.com/zpzim/STOMPSelfJoin) which has the following additional features:
 * Tiling for large inputs 
 * Computation in fp32 or fp64 (fp32 is recommended for most datasets and is much faster)
 * fp32 version should be compatible with GeForce cards
 * AB joins (you can produce the matrix profile from 2 different time series)
 * Distributable (we use AWS but other cloud platforms can work)
 * Some optimizations for Pascal.

Note: for self-joins on small inputs (~2M or less) the features in this repository are probably overkill. You should probably use [GPU-STOMP](https://github.com/zpzim/STOMPSelfJoin)

# Environment
This base project requires:
 * At least version 9.0 of the CUDA toolkit available [here](https://developer.nvidia.com/cuda-toolkit).
 * An NVIDIA GPU with CUDA support is also required. You can find a list of CUDA compatible GPUs [here](https://developer.nvidia.com/cuda-gpus)
 * Currently builds under linux with the Makefile. 
 * Should compile under windows, but untested. You will probably have to handle the parsing of command line arguments differently in windows.
 * We highly recommend using a Volta GPU (we get about a 3x improvement over Pascal)
# Usage
* Edit the Makefile (src/Makefile)
  * Volta and Pascal are supported by default, but if needed set the value of ARCH to correspond to the compute capability of your GPU.
    * "-gencode=arch=compute_code,code=sm_code" where code corresponds to the compute capability or arch you wish to add.
  * Make sure CUDA_DIRECTORY corresponds to the location where cuda is installed on your system. This is usually `/usr/local/cuda-(VERSION)/` on linux
* `(cd src && make)`
* `src/SCRIMP-GPU window_size input_A_file_path output_matrix_profile_path output_index_path`
  * Optional Arguments:
    * "-b [input B name]": allows a second input file which acts as the second time series for an AB join. An AB join compares every subsequence in input A with every subsequence in input B, the length of the matrix profile produced by this operation is always determined by input A, but the matrix profile index's values will reference subsequences in input B.
    * "-s [max tile size]": allows you to specify the max tile size used by the SCRIMP tile scheme. By default this is set to 2M, but you can adjust this as desired. Note that a tile size smaller than ~1M will likely fail to saturate the compute resources of newer GPUs
    * "-d": forces SCRIMP to compute the result in double precision. This is about 2x slower than single precision. This is uncessessary for most datasets, but if your input massively fluctuates in a wide range you may need it, see test/SampleInput/earthquake_precision_test.txt for an example of a dataset that fails in single precision.
    * "-g [device number to use]": allows you to specify which gpus to use on the machine, by default we try to use all of them. The device numbers must be valid cuda devices on your system. You can chain these to add more gpus. Example: -g 0 -g 1 will use gpu 0 and 1 on the system.
    * "-f [secondary output prefix]": only set this in a distributed environment. This forces AB joins to output a second matrix profile and index file which correspond to the reverse "BA" join. This is used to compute independant pieces of a large self-join that is distributed in separate jobs
    * "-r [global tile row]" : allows you to specify the tile row where this join starts, assuming that it is part of some larger join, this is required if you use -f
    * "-c [global tile col]" : allows you to specify the tile column where this join starts, assuming that it is part of some larger join, this is required if you use -f
* By default, if no devices are specified, SCRIMP will run on all available devices

# AWS operation
* This framework can be used with [AWS Batch](https://aws.amazon.com/batch) to distribute the computation to a cluster of p2 or p3 instances
* Information forthcoming

