Environment
===========

Currently builds under Windows/Mac/Linux using msvc/gcc/clang and nvcc (if CUDA is available) with cmake (3.8+ for cuda support)

Base dependancies (required for all builds of SCAMP):
  * cmake 3.8 or greater
  
    * This version is not available directly from all package managers so you may need to install it manually, the easist way to do this is with python via ``pip install cmake`` or you can download it manually from `here <https://cmake.org/download/>`_
 

For GPU support (required for any SCAMP build which will use a GPU):
  * cuda toolkit v9.0 or greater

    * Available `here <https://developer.nvidia.com/cuda-toolkit>`_ 

  * NVIDIA GPU with CUDA (compute capability 3.0+) support.

    * You can find a list of CUDA compatible GPUs `here <https://developer.nvidia.com/cuda-gpus>`_
    * Currently Supports Kepler-Turing, but Ampere and beyond will likely work as well, just add the -gencode flag for your specific architecture in CMakeLists.txt
    * Highly recommend using a Pascal/Volta GPU as they are much better (V100 is ~10x faster than a K80 for SCAMP, V100 is ~2-3x faster than a P100)

 
For python support:
  * Only python 3 is supported
  * Python 2 can work, but will not be supported if things break

Recommended Compiler:
 * If you are using CPUs, using clang v6.0 or above is highly recomended as gcc may not properly autovectorize the CPU kernels.


Notes on GPU Support
""""""""""""""""""""

You need to have a cuda development environment set up in order to build SCAMP with GPU support. If you install SCAMP (or pyscamp) and it does not detect CUDA during installation it will install using CPU support only. cmake must detect your cuda installation, this can be especially tricky when using Windows and MSVC as you need to have the CUDA extensions for visual studio installed. 

You can use the :ref:`configuration option <build-config-options>` FORCE_CUDA=1, to force SCAMP to build with CUDA (or fail). This currently only works on the CLI version, but will be added to the python module soon.



