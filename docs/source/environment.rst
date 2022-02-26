Environment
===========

Currently builds under Windows/Mac/Linux using msvc/gcc/clang and nvcc (if CUDA is available) with cmake (3.8+ for cuda support)

Base dependancies (required for all builds of SCAMP):
  * cmake 3.8 or greater (3.15 for pyscamp)
  
    * This version is not available directly from all package managers so you may need to install it manually, the easist way to do this is with python via ``pip install cmake`` or you can download it manually from `here <https://cmake.org/download/>`_

  * C/C++ compiler (e.g. gcc/clang/Visual Studio Build tools)
 
For GPU support (required for any SCAMP build which will use a GPU):
  * cuda toolkit v9.0 or greater

    * Available `here <https://developer.nvidia.com/cuda-toolkit>`_ 

  * NVIDIA GPU with CUDA (compute capability 3.0+) support.

    * You can find a list of CUDA compatible GPUs `here <https://developer.nvidia.com/cuda-gpus>`_
    * Highly recommend using a Pascal/Volta GPU as they are much better (V100 is ~10x faster than a K80 for SCAMP, V100 is ~2-3x faster than a P100)

 
For python support:
  * Only Python 3 is supported.

Recommended Compiler:
 * If you are using CPUs, using a newer version of clang is recommended as it tends to have better performance.


Notes on GPU Support
""""""""""""""""""""

You need to have a cuda development environment set up in order to build SCAMP with GPU support. If you install SCAMP (or pyscamp) and it does not detect CUDA during installation it will install using CPU support only. cmake must detect your cuda installation, this can be especially tricky when using Windows and MSVC as you need to have the CUDA extensions for visual studio installed. 

Windows CUDA builds will only work using visual studio tools (and the CUDA visual studio extensions). This is due to the fact that the visual studio toolchain is the only suppored toolchain for compiling cuda on windows, changing the C++ compiler will cause nvcc to fail. Therefore you can only use other generators for C++ only builds.

You can use the :ref:`configuration option <build-config-options>` FORCE_CUDA=1, to force SCAMP to build with CUDA (or fail). This works when installing pyscamp as well using ``FORCE_CUDA=1 pip install pyscamp``.



