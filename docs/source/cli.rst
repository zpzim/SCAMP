
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



