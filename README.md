# GPU-SCRIMP
This is a GPU implementation of the SCRIMP algorithm. SCRIMP takes a time series as input and computes the matrix profile for a particular window size. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
This is a more fleshed out implementation of [GPU-STOMP](https://github.com/STOMPSelfJoin) which has the following additional features:
 * Tiling for large inputs 
 * Computation in fp32 or fp64 (fp32 is recommended for most datasets and is much faster)
 * fp32 version should be compatible with GeForce cards
 * AB joins (you can produce the matrix profile from 2 different time series)
 * Distributable (we use AWS but other cloud platforms can work)
 * Some optimizations for Pascal.

Note: for self-joins on small inputs (~2M or less) the features in this repository are probably overkill. You should probably use [GPU-STOMP](https://github.com/STOMPSelfJoin)

# Environment
This base project requires:
 * At least version 9.0 of the CUDA toolkit available [here](https://developer.nvidia.com/cuda-toolkit).
 * An NVIDIA GPU with CUDA support is also required. You can find a list of CUDA compatible GPUs [here](https://developer.nvidia.com/cuda-gpus)
 * Currently builds under linux with the Makefile. 
 * Should compile under windows, but untested. 
 * We highly recommend using a Volta GPU (we get about a 3x improvement over Pascal)
# Usage
* Edit the Makefile (src/Makefile)
  * Volta and Pascal are supported by default, but if needed set the value of ARCH to correspond to the compute capability of your GPU.
    * "-gencode=arch=compute_code,code=sm_code" where code corresponds to the compute capability or arch you wish to add.
  * Make sure CUDA_DIRECTORY corresponds to the location where cuda is installed on your system. This is usually `/usr/local/cuda-(VERSION)/` on linux
* `(cd src && make)`
* `src/SCRIMP-GPU window_size max_tile_size fp64_flag full_join_override_flag input_A_file_path input_B_file_path output_matrix_profile_path output_indexes_path [Full join override: matrix profile and index output] (Optional: list of device numbers that you want to run on)`
  * "fp64 = 0 -> fp32 operation, otherwise fp64"
  * "full_join_override flag": this should be set to zero in most standard use cases, only set this in a distributed environment (documentation forthcoming)
* By default, if no devices are specified, SCRIMP will run on all available devices

# AWS operation
* This framework can be used with [AWS Batch](https://aws.amazon.com/batch) to distribute the computation to a cluster of p2 or p3 instances
* Information forthcoming

