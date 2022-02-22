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
    --profile_type=ALL_NEIGHBORS --max_matches_per_column=[K] \
    --threshold=[optional, correlation threshold]

This will generate one file: mp_columns out, which contains the all matches discovered as [matrix col (subseq index), matrix row (match index), distance] tuples.

For a matrix summary::

  ./SCAMP --window=window_size --input_a_file_name=input_A_file_path \
    --profile_type=MATRIX_SUMMARY --reduced_height=[H] --reduced_width=[W] \
    --threshold=[optional, correlation threshold for matches]

This will generate one file: mp_columns_out, which contains the matrix summary.

Input Format
************

Input files should have one value per line. The parser expects newlines at the end of each value. You can find examples of this on github in the `test/SampleInput <https://github.com/zpzim/SCAMP/tree/master/test/SampleInput>`_ directory.


Required Arguments
******************

``--input_a_file_name=[/path/to/file]``:
  The time series input to compute the matrix profile for, the input files should follow the format listed above.
 
``--window=[subsequence length]``:
  The subsequence length to use for computing the matrix profile.

Common Optional Arguments
*************************

``--gpus=[list of device numbers to use]``: 
  Allows you to specify which gpus to use on the machine, by default we try to use all of them. The device numbers must be valid cuda devices on your system. You can chain these to add more gpus. Example: ``--gpus="0 1"`` will use gpu 0 and gpu 1 on the system.
  
``--input_b_file_name=[/path/to/file]``:
  Allows a second input file which acts as the second time series for an AB join. An AB join compares every subsequence in input A with every subsequence in input B, the length of the matrix profile produced by this operation is always determined by input A, but the matrix profile index's values will reference subsequences in input B. Providing this parameter implies that SCAMP will compute an AB join.  

``--no_gpu``:
  SCAMP will not use GPUs even if they are available. Must be combined with ``--num_cpu_workers`` or no computation will happen.

``--num_cpu_workers=[number of threads]``:
  Allows you to specify the number of cpu threads to compute with, by default we use none. For now, if you don't have gpus, we recommend setting this to the number of cores on your system for best performance. It is possible to perform hetrogeneous GPU/CPU computation using this flag, but you will likely see very little speedup compared to GPU only as GPUs are much faster. If you have GPUs on your system and don't want to use them you should also use the ``--no_gpu`` argument.
  
``--output_a_file_name=[/path/to/output/file]``:
  This is the name of the output file where SCAMP will write its output (except index values in 1NN_INDEX profiles) [default: mp_columns_out]

``--output_a_index_file_name=[/path/to/output/file]``:
  This is the name of the output file where SCAMP will write the indexes for 1NN_INDEX profiles [default: mp_columns_out_index]

``--output_b_file_name=[/path/to/output/file]``:
  Only used when ``--keep_rows`` is specified, this is the name of the file where SCAMP will write the row-wise matrix profile [default: mp_rows_out]
 
``--output_b_index_file_name=[/path/to/output/file]``:
  Only used when ``--keep_rows`` is specified, this is the name of the file where SCAMP will write the row-wise matrix profile indexes [default: mp_rows_out_index]
  
``--output_pearson``:
  SCAMP will output pearson correlation rather than z-normalized euclidean distance.

``--print_debug_info``:
  By default SCAMP runs in silent_mode with no output, this option prints debugging information to stdout.

``--profile_type=[Type of profile to compute]``:
  Determines the type of matrix profile to compute. See the examples above and in :doc:`Profile Types </profiles>`.

``--reduced_height=[height of output matrix]``:
  For matrix summary profiles, the height of the reduced resoulution distance matrix output.

``--reduced_width=[width of output matrix]``:
  For matrix summary profiles, the width of the reduced resoulution distance matrix output.

``--threshold=[correlation threshold in the interval [0,1] ]``:
  For sum / histogram / knn / matrix_summary profiles. Correlations below this value will be ignored from the final output.

Advanced Optional Arguments
***************************

One of [``--ultra_precision``, ``--double_precision``, ``--mixed_precision``, ``--single_precision``]:
  Changes the precision mode of SCAMP, default is double precision (and is the only one available on CPU joins), mixed precision will work on many datasets but not all, single precision will work for some simple datasets, but may prove unreliable for many. See test/SampleInput/earthquake_precision_test.txt for an example of a dataset that fails in mixed/single precision. The single precision mode is about 2x faster than double precision, mixed_precision falls in the middle, but can sometimes be as slow as double precision".

  Ultra Precision uses double precision everywhere and also computes the norms of each subsequence during the precomputation step with a more precise, but potentially slower formula with complexity **O(len(timeseries)*sublen)**, this should be tried if you run into issues with the standard method in double precison and need more precise calculations. Also uses a new, more stable formula for computing the matrix profile. 

``--aligned``:
  For ab-joins only, indicates that A and B may start with the same sequence and must consider an exclusion zone.


``--keep_rows``:
  Informs SCAMP to compute the "rowwise mp" and output in a a separate file specified by the flag ``--output_b_file_name``.

  #. In self-joins, specifying this flag results in "output_a_file_name" containing the right matrix profile and "output_b_file_name" containing the left, these can be used to compute time series chains. 

  #. This is also useful when computing a distributed self-join, so as to not recompute values in the lower-trianglular portion of the symmetric distance matrix.

``--global_col=[global col of the distance matrix in a distributed join]``:
  Informs SCAMP that this join is part of a larger distributed join which starts at this column in the larger distance matrix, this allows us to pick an appropriate exclusion zone for our computation if necessary.

``--global_row=[global row of the distance matrix in a distributed join]``:
  Informs SCAMP that this join is part of a larger distributed join which starts at this row in the larger distance matrix, this allows us to pick an appropriate exclusion zone for our computation if necessary.

``--max_tile_size=[integer tile size]``:
  Allows you to specify the max tile size used by the SCAMP tile scheme. By default this is set to 128K, but you can adjust this as desired. Note that a tile size smaller than 128K will likely fail to saturate the compute resources of newer GPUs


.. _build-config-options:

Build Configuration Options
***************************


Specifying a Compliler
************************************

On Linux or Mac, if you need to specify a specific compiler or cuda toolkit if you have multiple installed, you can use the following defines. By default cmake will look for cuda at the /usr/local/cuda symlink on linux::

  cmake -DCUDACXX=/path/to/nvcc \
        -DCXX=/path/to/cpp/compiler \
        -DCC=/path/to/c/compiler ..

On Windows this is slightly different as you need to specify the generator to cmake::

  # Build with Visual Studio 2015 tools
  cmake -G "Visual Studio 14 2015" ..
  # Build with Ninja (requires ninja)
  cmake -G "Ninja" -DCMAKE_CXX_COMPILER=/path/to/compiler

Windows CUDA builds will only work using Visual Studio tools (and the CUDA visual studio extensions). This is due to the fact that the visual studio toolchain is the only suppored toolchain for compiling cuda on windows, changing the C++ compiler will cause nvcc to fail. Therefore you can only use other generators for C++ only builds.

Forcing CUDA (or No CUDA)
************************************

If you desire explicit CUDA support, you can make the build fail using the flag FORCE_CUDA=1 if cuda is not found::
  
  cmake -DFORCE_CUDA=1 ..

The same is true if you want to disable CUDA support using FORCE_NO_CUDA=1, this will cause CUDA not to be used, even if it is found on the system::

  cmake -DFORCE_NO_CUDA=1 ..



