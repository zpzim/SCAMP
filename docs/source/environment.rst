Environment
===========

Currently builds under Windows/Mac/Linux using msvc/gcc/clang and nvcc (if CUDA is available) with cmake.

Base dependancies (required for all builds of SCAMP):
  * cmake 3.18 or greater
  
    * This version is not available directly from all package managers so you may need to install it manually, the easist way to do this is with python via ``pip install cmake`` or you can download it manually from `here <https://cmake.org/download/>`_

  * C/C++ compiler (e.g. gcc/clang/Visual Studio Build tools)

  * SCAMP is only tested currently on x86_64 systems. 32-bit systems are not supported. Though SCAMP may build on them, other 64-bit architectures are not currently tested or optimized for.

 
For GPU support (required for any SCAMP build which will use a GPU):
  * cuda toolkit v11.0 or greater

    * Available `here <https://developer.nvidia.com/cuda-toolkit>`_ 

  * NVIDIA GPU with CUDA (compute capability 3.5+) support.

    * You can find a list of CUDA compatible GPUs `here <https://developer.nvidia.com/cuda-gpus>`_
    * Highly recommend using a Pascal/Volta or newer GPU as they are much better (V100 is ~10x faster than a K80 for SCAMP, V100 is ~2-3x faster than a P100)

 
For python support:
  * Only Python 3 is supported.

Recommended Compiler:
 * If you are using CPUs, using a newer version of clang is recommended as it tends to have better performance.


Notes on GPU Support
""""""""""""""""""""

You need to have a cuda development environment set up in order to build SCAMP with GPU support. If you install SCAMP (or pyscamp) and it does not detect CUDA during installation it will install using CPU support only. cmake must detect your cuda installation, this can be especially tricky when using Windows and MSVC as you need to have the CUDA extensions for visual studio installed.

I have only gotten Windows CUDA builds to work under MSVC and the Visual Studio Generators. There are some issues with cmake/nvcc/msvc that make it very difficult to install outside of this configuration.

You can use the :ref:`configuration option <build-config-options>` FORCE_CUDA=1, to force SCAMP to build with CUDA (or fail). This works when installing pyscamp as well using ``FORCE_CUDA=1 pip install pyscamp``.


Environment variables
"""""""""""""""""""""

SCAMP reads a handful of environment variables, both at build time and
at run time. The build-time ones map to CMake cache variables of the
same name; the run-time ones tune behavior of the CLI and pyscamp.
For convenience, all of them are listed here in one place.

**Build-time variables** (read by ``cmake`` / ``setup.py`` at configure
time):

* ``FORCE_CUDA`` — fail the build if CUDA isn't detected, instead of
  silently dropping to a CPU-only build. Useful for catching missing
  CUDA installations during pip installs of pyscamp.
* ``FORCE_NO_CUDA`` — force a CPU-only build even if CUDA is detected
  on the system.
* ``CMAKE_CUDA_ARCHITECTURES`` — comma-separated list of SM architectures
  to compile for (e.g. ``86`` for a single dev GPU, or
  ``75;80;86;89;90`` for a redistributable build). Overrides the
  per-CUDA-version default set in ``cmake/SCAMPMacros.cmake``. For
  local-dev builds against a single known GPU, setting this to your
  device's SM number alone cuts compile time substantially.
* ``BUILD_CLIENT_SERVER`` — enable the gRPC distributed worker /
  driver targets (see :doc:`distributed`).
* ``BUILD_PYTHON_MODULE`` — build the pyscamp bindings.
* ``SCAMP_ENABLE_BINARY_DISTRIBUTION`` — for binary wheel / conda-forge
  builds; tells the build to bake in defaults appropriate for
  redistribution rather than for the developer's local box.
* ``SCAMP_USE_CLANG_TIDY`` — run clang-tidy on the SCAMP sources during
  the build. Off by default.

A few additional pyscamp-specific build-time knobs (``PYSCAMP_PYTHON_EXECUTABLE_PATH``,
``PYSCAMP_ADD_CMAKE_ARGS``, ``PYSCAMP_BUILD_TYPE``,
``PYSCAMP_NO_PLATFORM_AUTOSELECT``, ``PYSCAMP_USE_EXTERNAL_PYBIND11``)
are read by ``setup.py`` when building the Python bindings; see
:doc:`pyscamp/intro` for details.

**Run-time variables** (read by the SCAMP binary, pyscamp, or the
distributed gRPC client):

* ``SCAMP_AUTOTUNE_CACHE`` — explicit path to read/write the autotune
  cache from. Overrides the platform-default location (see
  :ref:`autotune-default-path`).
* ``XDG_CACHE_HOME`` — when set, SCAMP's autotune cache lives under
  ``$XDG_CACHE_HOME/scamp/autotune.txt`` (any platform).
* ``HOME`` (Linux/macOS), ``LOCALAPPDATA`` / ``USERPROFILE`` (Windows)
  — used to derive the platform-default autotune cache path when
  neither ``SCAMP_AUTOTUNE_CACHE`` nor ``XDG_CACHE_HOME`` is set.
  See :ref:`autotune-default-path`.
* ``SCAMP_AUTOTUNE_QUIET`` — set to ``1`` (or any truthy non-empty
  value other than ``0`` / ``false`` / ``FALSE``) to suppress the
  one-shot "no autotune entry for device …" warning on cache miss.
  Set by pyscamp at module init for notebook users; the CLI leaves it
  off by default. See :ref:`autotune-miss-warning`.
* ``SCAMP_AUTOTUNE_INPUT_LENGTH`` — synthetic input length the
  ``--autotune`` benchmark uses per trial (default 131072 = 128K
  elements). Larger values are slower but produce per-variant rankings
  that match production-scale workloads better; see :doc:`autotune` for
  guidance on choosing a value.
* ``SCAMP_SERVER_SERVICE_HOST`` and ``SCAMP_SERVER_SERVICE_PORT`` —
  host and port the gRPC ``SCAMPclient`` connects to. Only relevant
  for the distributed worker / driver build (see :doc:`distributed`).



