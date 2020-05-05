.. highlight:: console

SCAMP CLI
=========

SCAMP provides a command line interface which can be used to generate matrix profiles from ascii files. The command line interface provides the most flexibility in terms of how you can execute ``SCAMP``, but ``pyscamp``, is likely easier to interact with programmatically.

Building the SCAMP CLI
**********************


Clone the repository and submodules; make a build directory::

  git clone https://github.com/zpzim/SCAMP
  cd SCAMP
  git submodule update --init --recursive
  mkdir build && cd build
  
Build SCAMP on Mac/Linux/Windows::

  # cmake will look in your $PATH for the cuda/c++ compilers
  # If you have problems with cmake, you may need to specify a
  # cuda or c++ compiler as shown above
  cmake ..
  cmake --build . --config Release


Using the SCAMP CLI
*******************

Once built, you can use the CLI as follows

For a traditional matrix profile::

  ./SCAMP --window=window_size --input_a_file_name=input_A_file_path

This will generate two files: mp_columns_out and mp_columns_out_index, which contain the matrix profile and matrix profile index values respectively. 

For a sum/histogram profile::

  ./SCAMP --window=window_size --input_a_file_name=input_A_file_path \
     --profile_type=SUM_THRESH \
     --threshold=[optional, correlation threshold for matches]

This will generate one file: mp_columns_out, which contains histogram matrix profile (in sum of correlations above threshold). 

For a knn matrix profile::

  ./SCAMP --window=window_size --input_a_file_name=input_A_file_path \
    --profile_type=ALL_NEIGHBORS --max_matcher_per_column=[K] \
    --threshold=[optional, correlation threshold]

This will generate one file: mp_columns out, which contains the all matches discovered as [matrix col (subseq index), matrix row (match index), distance] tuples.

For a matrix summary::

  ./SCAMP --window=window_size --input_a_file_name=input_A_file_path \
    --profile_type=MATRIX_SUMMARY --reduced_height=[H] --reduced_width=[W] \
    --threshold=[optional, correlation threshold for matches]

This will generate one file: mp_columns_out, which contains the matrix summary.


Optional Arguments
******************

The above examples can use the following optional arguments (and more):

  * "--input_b_file_name=/path/to/file": allows a second input file which acts as the second time series for an AB join. An AB join compares every subsequence in input A with every subsequence in input B, the length of the matrix profile produced by this operation is always determined by input A, but the matrix profile index's values will reference subsequences in input B. Providing this parameter implies that SCAMP will compute an AB join.
  * "--max_tile_size=[integer tile size]": allows you to specify the max tile size used by the SCAMP tile scheme. By default this is set to 128K, but you can adjust this as desired. Note that a tile size smaller than 128K will likely fail to saturate the compute resources of newer GPUs
  * One of "--double_precision, --mixed_precision, --single_precision": Changes the precision mode of SCAMP, default is double precision (and is the only one available on CPU joins), mixed precision will work on many datasets but not all, single precision will work for some simple datasets, but may prove unreliable for many. See test/SampleInput/earthquake_precision_test.txt for an example of a dataset that fails in mixed/single precision. The single precision mode is about 2x faster than double precision, mixed_precision falls in the middle, but can sometimes be as slow as double precision".
  * "--gpus=\"list of device numbers to use\"": allows you to specify which gpus to use on the machine, by default we try to use all of them. The device numbers must be valid cuda devices on your system. You can chain these to add more gpus. Example: --gpus="0 1" will use gpu 0 and gpu 1 on the system.
  * "--num_cpu_workers": allows you to specify the number of cpu threads to compute with, by default we use none. For now, if you don't have gpus, we recommend setting this to the number of cores on your system for best performance. It is possible to perform hetrogeneous GPU/CPU computation using this flag, but you will likely see very little speedup compared to GPU only as GPUs are much faster. 

There are more arguments that allow you even greater control over what SCAMP can do. Use --helpfull for a list of possible arguments and their descriptions.


.. _build-config-options:

Build Configuration Options
***************************


Specifying a Compliler
************************************

On Linux or Mac, if you need to specify a specific compiler or cuda toolkit if you have multiple installed, you can use the following defines. By default cmake will look for cuda at the /usr/local/cuda symlink on linux::

  cmake -D CMAKE_CUDA_COMPILER=/path/to/nvcc \
        -D CMAKE_CXX_COMPILER=/path/to/cpp/compiler \
        -D CMAKE_C_COMPILER=/path/to/c/compiler ..

On Windows this is slightly different as you need to specify the generator to cmake::

  # Build with Visual Studio 2015 tools
  cmake -G "Visual Studio 14 2015" ..
  # Build with Ninja (requires ninja)
  cmake -G "Ninja" -DCMAKE_CXX_COMPILER=/path/to/compiler

Windows CUDA builds seem to only work using visual studio tools (and the CUDA visual studio extensions) currently. Looking into making this work more generally with other generators.

Forcing CUDA (or No CUDA)
************************************

If you desire explicit CUDA support, you can make the build fail using the flag FORCE_CUDA=1 if cuda is not found::
  
  cmake -D FORCE_CUDA=1 ..

The same is true if you want to disable CUDA support using FORCE_NO_CUDA=1, this will cause CUDA not to be used, even if it is found on the system::

  cmake -D FORCE_NO_CUDA=1 ..



