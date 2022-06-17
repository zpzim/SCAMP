pyscamp
=======

``pyscamp`` is a python module which uses the SCAMP CUDA/C++ library to compute the matrix profile efficiently, inheriting the speed and functionality of SCAMP.


Installation
------------
There are two ways to install pyscamp, via its conda package, or directly from source.

Installing on conda-forge
^^^^^^^^^^^^^^^^^^^^^^^^^
In an anaconda environment, you can install the pyscamp conda packages as follows:

To install pyscamp with gpu support (Windows/Linux) ``conda install -c conda-forge pyscamp-gpu``.

To install pyscamp without gpu support (Windows/Linux/MacOS) ``conda install -c conda-forge pyscamp-cpu``.

Note that ``pyscamp-gpu`` can be installed and used even if you don't have a GPU, it will simply fall back to using your CPU. However, ``pyscamp-cpu`` is preferrable if you don't have a GPU because it builds with a newer compiler and does not require installing the ``cudatoolkit`` depencency.

If you run into problems using GPUs with ``pyscamp-gpu`` make sure your NVIDIA drivers are up to date. This is the most common cause of issues.

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
You can install pyscamp from source to maximize performance on your system. A source distribution for pyscamp is available on pypi.org.

The python module supports python3 and requires that cmake 3.15 or greater is installed in order to build properly.

You can install cmake using ``pip install cmake``

If you want GPU support for pyscamp, you must have CUDA installed and available to the default cmake compiler on your system. 

Please look at the :doc:`environment setup guide </environment>` for more information about how to set up the environment for SCAMP.

Once you have cmake and/or cuda installed, you can use ``pip install pyscamp`` to download and install the module.

pyscamp allows you to specify certain build options to pip using environment variables:

  * ``FORCE_CUDA=1`` will allow you to force pyscamp to build with cuda (or fail if it can't find CUDA)
  * ``FORCE_NO_CUDA=1`` will force pyscamp to build without cuda, even if it is found on the system.
  * ``CC=<compiler>`` will force pyscamp to use a c compiler other than the default compiler detected by cmake, this works only for builds not using visual studio tools.
  * ``CXX=<compiler>`` will force pyscamp to use a c++ compiler other than the default compiler detected by cmake, this works only for builds not using visual studio tools
  * ``CUDACXX=<nvcc location>`` will point pyscamp to the specified cuda compiler to build with, useful if your cuda compiler is not detected. This works only for builds not using visual studio tools
  * ``CMAKE_GENERATOR=<generator tag ("Ninja", etc.)>`` useful for windows builds to specify a generator other than visual studio tools
  * ``CMAKE_GENERATOR_TOOLSET=<toolset tag ("llvm", etc.)>`` for specifying a non-default compiler toolchain when using visual studio tools
  * ``CMAKE_GENERATOR_PLATFORM=<platform tag (x64)>`` for specifying a non-default platform arch when using visual studio tools.
  * ``PYSCAMP_USE_EXTERNAL_PYBIND11=ON`` pyscamp will install using the pybind11 package installed on the system. Used to build the conda-forge package.
  * ``PYSCAMP_PYTHON_EXECUTABLE_PATH=<Path to the python executable to build for>`` defaults to the execuatable invoking setup.py. If you want to build pyscamp for a specific python executable you can point this env variable to it.
  * ``PYSCAMP_ADD_CMAKE_ARGS=<args>`` Allows for passing additional cmake arguments during pyscamp's build.

.. highlight:: console

Example::

  # Force a cuda build using clang++ (on the system path) as the cpp compiler
  FORCE_CUDA=1 CC=clang CXX=clang++ pip install pyscamp

**Installation Notes**: If you do not specify any of the above options, cmake will use the default compilers available to it, will try to find cuda and install GPU support if it is found.


Installation Troubleshooting
----------------------------

If you have problems building the module (or getting GPU support to work) please submit an issue on github. I don't have access to all build environments so help in addressing these issues is appreciated.

**Figuring out what went wrong**: You can use ``pip install -v pyscamp`` to print the output of the cmake configuration and build.

**Getting CUDA to work**: 

**When installing from conda-forge**: If you have installed the ``pyscamp-gpu`` conda-forge package and you are having trouble with CUDA the most common issue is that the NVIDIA drivers on the system need to be updated to work with the newest versions of CUDA. Please try to update your GPU drivers.

**When building from source**: Ensure pyscamp is built with cuda using ``FORCE_CUDA=1 pip install -I --no-cache-dir pyscamp``. If this fails, that means cmake was unable to detect your cuda installation or it wasn't new enough (see :doc:`environment setup guide </environment>` for which versions of cuda are supported). Some general troubleshooting steps you can try are:

  * If you installed pyscamp previously and you have since installed cuda, make sure to add the ``-I`` and ``--no-cache-dir`` flags to pip install just to make sure you are reinstalling correctly.
  * Use ``pip install -v`` to get more information about the build configuration and make sure it is using the compilers and cuda like you expect.
  * On Mac/Linux make sure nvcc (the CUDA compiler, usually located at /usr/local/cuda/bin), is in your PATH. You can also specify a cuda compiler using ``CUDACXX=/path/to/cuda/compiler pip install pyscamp``
  * On Windows, I have only gotten CUDA to work using using the visual studio toochains **with the appropriate visual studio plugins installed** so make sure cuda is installed with these plugins. (see :doc:`GPU support </environment>` for more information and links to the cuda installation guide)

    * This means that it is not currently possible to use a compiler other than MSVC to build SCAMP with CUDA support on Windows.

**Using a different compiler**:

  * On Mac/Linux: You can install clang v6 or greater and point pyscamp to it using ``CXX=path/to/compiler pip install pyscamp``
  * On Windows: You can use Ninja (or another generator) to build with ``CMAKE_GENERATOR=Ninja CXX=path/to/compiler pip install pyscamp``

pyscamp System Resource Usage
-----------------------------

When a pyscamp method is invoked with the default arguments. The following logic is followed to determine how to use resources on the system:

 1. Check if GPUs are available, if so use them, do not use CPU resources to do compute heavy work.
 2. If GPUs are not available, pyscamp will use cpu threads equal to the number of available cores to do compute work.

This logic is followed by default, but can be changed with the ``gpus`` and ``threads`` pyscamp kwargs:

 * If you want to opt out of gpu execution, specify an empty list e.g. ``gpus=[]``.
 * If you want to use a non-default number of threads, specify the number in ``threads=N``. Note that this is not recommended when GPUs are being used by default, so you should also specify ``gpus=[]`` so that you don't mix CPU/gpu resources. The only exception to this is if you want to use all resources available to compute results on a very large input. Otherwise, mixing cpu/gpu resources will probably end up slower than simply using GPU resources alone.

Python Example
--------------

.. highlight:: python

::

  import pyscamp as mp

  # Allows checking if pyscamp was built with CUDA and GPUs are available.
  has_gpu_support = mp.gpu_supported()

  # Self join.
  profile, index = mp.selfjoin(a, sublen)
  # AB join using 4 threads and no gpus.
  profile, index = mp.abjoin(a, b, sublen, threads=4, gpus=[])
  # Sum thresh
  corr_sum = mp.abjoin_sum(a, b, sublen, threshold=0.9)
    
  # Matrix summary (100x100) with threVshold, outputting pearson correlation
  matrix = mp.abjoin_matrix(a, b, sublen, mwidth=100, mheight=100, threshold=0.5, pearson=True)

  # Approximate KNN is supported with GPUs + CUDA only for now.
  if has_gpu_support:
    knn = mp.selfjoin_knn(a,sublen, k)
    # KNN with threshold
    knn = mp.selfjoin_knn(a, sublen, k, threshold=0.85)
    # KNN Ab join with threshold, outputting pearson correlation
    knn = mp.abjoin_knn(a, b, sublen, k, threshold=0.90, pearson=True)


