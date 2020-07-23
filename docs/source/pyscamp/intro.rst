pyscamp
=======

pyscamp is a python module which uses the SCAMP CUDA/C++ library to compute the matrix profile efficiently, inheriting the speed and functionality of SCAMP


Installation
------------
A source distribution for pyscamp is available on pypi.org. The python module supports python 3 and requires that cmake is installed in order to build properly, you can install cmake using ``pip install cmake``

If you want GPU support for pyscamp, you must have CUDA installed and available to the default cmake compiler on your system. 

Please look at the :doc:`environment setup guide </environment>` for more information about how to set up the environment for SCAMP.

Once you have cmake and/or cuda installed, you can use ``pip install pyscamp`` to download and install the module.

pyscamp allows you to specify certain build options to pip using environment variables:

  * ``FORCE_CUDA=1`` will allow you to force pyscamp to build with cuda (or fail if it can't find CUDA)
  * ``FORCE_NO_CUDA=1`` will force pyscamp to build without cuda, even if it is found on the system.
  * ``CMAKE_CXX_COMPILER=<compiler>`` will force pyscamp to use a compiler other than the default compiler detected by cmake, this works only for builds not using visual studio tools
  * ``CMAKE_CUDA_COMPILER=<nvcc location>`` will point pyscamp to the specified cuda compiler to build with, this works only for builds not using visual studio tools
  * ``CMAKE_GENERATOR=<generator tag ("Ninja", etc.)>`` useful for windows builds to specify a generator other than visual studio tools
  * ``CMAKE_GENERATOR_TOOLSET=<toolset tag ("llvm", etc.)>`` for specifying a non-default compiler toolchain when using visual studio tools

.. highlight:: console

Example::

  # Force a cuda build using clang++ (on the system path) as the cpp compiler
  FORCE_CUDA=1 CMAKE_CXX_COMPILER=clang++ pip install pyscamp

**Installation Notes**: If you do not specify any of the above options, cmake will use the default compilers available to it, will try to find cuda and install GPU support if it is found.


Installation Troubleshooting
----------------------------

This is a new feature and still has some kinks to work out. If you have problems building the module (or getting GPU support to work) please submit an issue on github. I don't have access to all build environments so help in addressing these issues is appreciated.

**Figuring out what went wrong**: You can use ``pip install -v pyscamp`` to print the output of the cmake configuration and build.

**Getting CUDA to work**: Ensure pyscamp is built with cuda using ``FORCE_CUDA=1 pip install -I --no-cache-dir pyscamp`` if this fails. That means cmake was unable to detect your cuda installation or it wasn't new enough (see :doc:`environment setup guide </environment>` for which versions of cuda are supported). Some general troubleshooting steps you can try are:

  * If you installed pyscamp previously and you have since installed cuda, make sure to add the ``-I`` and ``--no-cache-dir`` flags to pip install just to make sure you are reinstalling correctly.
  * Use ``pip install -v`` to get more information about the build configuration and make sure it is using the compilers and cuda like you expect.
  * On Mac/Linux make sure nvcc (the CUDA compiler, usually located at /usr/local/cuda/bin), is in your PATH. You can also specify a cuda compiler using ``CMAKE_CUDA_COMPILER=/path/to/cuda/compiler pip install pyscamp``
  * On Windows, CUDA will only work using using the visual studio toochains **with the appropriate visual studio plugins installed** so make sure cuda is installed with these plugins. (see :doc:`GPU support </environment>` for more information and links to the cuda installation guide)

**Using a newer compiler for faster CPU Code**:

  * On Mac/Linux: You can install clang v6 or greater and point pyscamp to it using ``CMAKE_CXX_COMPILER=path/to/compiler pip install pyscamp``
  * On Windows: You can use Ninja to build with ``CMAKE_GENERATOR=Ninja CMAKE_CXX_COMPILER=path/to/compiler FORCE_NO_CUDA=1 pip install pyscamp``
  
    * CUDA **will not** work on Windows with a custom toolchain, MSVC + Visual Studio plugins for CUDA must be installed, so using the above method will not have CUDA support.



Python Example
--------------

.. highlight:: python

::

  import pyscamp as mp # Uses GPU if available and CUDA was available during the build

  # Allows checking if pyscamp was built with CUDA and has GPU support
  has_gpu_support = mp.gpu_supported()

  # Self join
  profile, index = mp.selfjoin(a, sublen)
  # AB join using 4 threads
  profile, index = mp.abjoin(a, b, sublen, threads=4)
  # Sum thresh
  corr_sum = mp.abjoin_sum(a, b, sublen, threshold=0.9)

  # Approximate KNN and matrix summaries are supported with GPUs + CUDA only
  if has_gpu_support:
    knn = mp.selfjoin_knn(a,sublen, k)
    # KNN with threshold
    knn = mp.selfjoin_knn(a, sublen, k, threshold=0.85)
    # KNN Ab join with threshold, outputting pearson correlation
    knn = mp.abjoin_knn(a, b, sublen, k, threshold=0.90, pearson=True)
    # Matrix summary (100x100) with threshold, outputting pearson correlation
    matrix = mp.abjoin_matrix(a, b, sublen, mwidth=100, mheight=100, threshold=0.5, pearson=True)


