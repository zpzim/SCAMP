cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
include(CheckLanguage)
include(CheckCXXCompilerFlag)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Determine if we are using a VS generator where the platform is specified in the name
# of the generator. We do not want to specify CMAKE_GENERATOR_PLATFORM in that case.
if (CMAKE_GENERATOR MATCHES "Visual Studio")
  if (CMAKE_GENERATOR MATCHES "Win64" OR CMAKE_GENERATOR MATCHES "ARM")
    set(USING_PLATFORM_SPECIFIC_VS_GENERATOR "True")
  endif()
endif()

if ((NOT CMAKE_GENERATOR) OR (NOT CMAKE_GENERATOR MATCHES "Visual Studio") OR USING_PLATFORM_SPECIFIC_VS_GENERATOR)
  # CMAKE_GENERATOR_PLATFORM is only needed when building with Visual Studio, and the
  # Visual Studio Generator is not configured by-name to already be building for a
  # particular platform, this generator-name-based selection of platform is only
  # supported on older versions of visual studio but some of these older versions are
  # still in wide use.
  # See: https://cmake.org/cmake/help/latest/generator/Visual%20Studio%2015%202017.html

  # Unfortunately, it is nearly impossible to determine what generator cmake will
  # use by default before invoking it, on Windows, cmake checks the registry to
  # figure out what generator to use.

  # Since we do not want to do 32-bit builds when 64-bit is available, checking and
  # removing the CMAKE_GENERATOR_PLATFORM variable if it does not apply allows for
  # us to set "-A x64" as an option to cmake regardless of the generator we are using.
  # This prevents some issues with building on Windows and makes it unnecessary for us
  # to determine what generator cmake will use before invoking it.

  # Note: this could break on other generators which make use of
  # CMAKE_GENERATOR_PLATFORM in the future.
  if (CMAKE_GENERATOR_PLATFORM OR ENV{CMAKE_GENERATOR_PLATFORM})
    unset(CMAKE_GENERATOR_PLATFORM CACHE)
    unset(ENV{CMAKE_GENERATOR_PLATFORM})
    message(WARNING "unset CMAKE_GENERATOR_PLATFORM because it is meaningless for generator ${CMAKE_GENERATOR}")
  endif()
endif()

# ----------------------------------------------------------------------------------------
#                              ===== Project Setup =====
project(SCAMPmain LANGUAGES CXX)
set(VERSION_MAJOR 1)
set(VERSION_MINOR 1)

set(CUDA_MINIMUM_VERSION "9.0")
include(SCAMPMacros)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
  message(WARNING "Detected a 32-bit system. 32-bit operation is unsupported and may not work correctly.")
endif()

# Thread libraries
find_package(Threads REQUIRED)

if(FORCE_NO_CUDA)
  unset(CMAKE_CUDA_COMPILER)
else()
  # Use cuda if available
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # For CUFFT libraries
    find_package(CUDA ${CMAKE_CUDA_VERSION} REQUIRED)
  endif()
endif()

# Do not use an unsupported cuda version
if (CMAKE_CUDA_COMPILER)
  if (CUDA_VERSION VERSION_LESS ${CUDA_MINIMUM_VERSION})
    if(FORCE_CUDA)
      message(FATAL_ERROR "CUDA version ${CUDA_VERSION} is less than the mininmum required version ${CUDA_MINIMUM_VERSION} defaulting to no-cuda build")
    else()
      message(WARNING "CUDA version ${CUDA_VERSION} is less than the mininmum required version ${CUDA_MINIMUM_VERSION} defaulting to no-cuda build")
      unset(CMAKE_CUDA_COMPILER)
    endif()
  endif()
endif()

if(NOT CMAKE_CUDA_COMPILER)
  if(FORCE_CUDA)
    message(FATAL_ERROR "No CUDA compiler found, cannot proceed to build CUDA binary")
  else()
    message(STATUS "No CUDA compiler found, building SCAMP without CUDA.")
  endif()
else()
  message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
  message(STATUS "Using cufft libraries: ${CUDA_CUFFT_LIBRARIES}")
endif()

fetch_env(SCAMP_ENABLE_BINARY_DISTRIBUTION)
fetch_env(BUILD_PYTHON_MODULE)
fetch_env(BUILD_CLIENT_SERVER)
fetch_env(SCAMP_USE_CLANG_TIDY)
fetch_env(BUILDING_PYSCAMP)

if (SCAMP_ENABLE_BINARY_DISTRIBUTION)
  message(STATUS "Building REDISTRIBUTABLE SCAMP Binary.")
endif()

if (SCAMP_USE_CLANG_TIDY)
  find_program(
    CLANG_TIDY_EXE
    NAMES clang-tidy
    DOC "Path to clang-tidy executable (v5+)"
    )
  if(NOT CLANG_TIDY_EXE)
    message(STATUS "clang-tidy not found.")
  else()
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-checks=*,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-fuchsia-default-arguments,-hicpp-vararg,-cppcoreguidelines-pro-type-vararg, -cppcoreguidelines-pro-bounds-constant-array-index,-google-runtime-int")
  endif()
else()
  message(STATUS "clang-tidy is not enabled. To use clang-tidy checks pass -DSCAMP_USE_CLANG_TIDY=ON to cmake")
endif()

# Use clang format if available
find_program(
  CLANG_FORMAT_EXE
  NAMES clang-format-6.0 clang-format-5.0 clang-format
  DOC "Path to clang-format executable (v5+)"
  )
if(NOT CLANG_FORMAT_EXE)
  message(STATUS "clang-format not found.")
else()
  message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")
  set(DO_CLANG_FORMAT "${CLANG_FORMAT}" "-i -style=file")
endif()

set(
  PROJECT_SOURCE_FILES
  src/core/*.cpp
  src/core/*.cu
  src/core/*.h
  src/core/cpu_kernel/*.cpp
  src/core/cpu_kernel/*.h
  src/core/cpu_kernel/baseline/*.cpp
  src/core/cpu_kernel/baseline/*.h
  src/core/cpu_kernel/avx/*.cpp
  src/core/cpu_kernel/avx/*.h
  src/core/cpu_kernel/avx2/*.cpp
  src/core/cpu_kernel/avx2/*.h
  src/common/*.cpp
  src/common/*.h
  src/distributed/*.cc
  src/distributed/*.cpp
  src/distributed/*.h
  src/python/*.cpp
  )

# ----------------------------------------------------------------------------------------
#                         ===== Compiler Configuration =====

set(CMAKE_CXX_STANDARD 17)

# CUDA_CONFIG
if (CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_STANDARD 14)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
  if (CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_80,code=sm_80")
  endif()
  if (CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_75,code=sm_75")
  endif()
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_70")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_61,code=sm_61")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_60")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_50,code=sm_50")
  if (CUDA_VERSION VERSION_LESS "11.0")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_37,code=sm_37")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_35,code=sm_35")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_30,code=sm_30")
  endif()
endif()

if (NOT SCAMP_ENABLE_BINARY_DISTRIBUTION)
  # AVX Config for MSVC
  find_package(AVX)
endif()

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-xHost" COMPILER_OPT_XHOST_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("/QxHost" COMPILER_OPT_QXHOST_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-O3" COMPILER_OPT_O3_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-funroll-loops" COMPILER_OPT_UNROLL_LOOPS_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-ffp-contract=fast" COMPILER_OPT_FPCONTRACT_FAST_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("/fp:contract" COMPILER_OPT_FPCONTRACT_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("/fp:precise" COMPILER_OPT_FPPRECISE_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-fPIC" COMPILER_OPT_PIC_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-Wall" COMPILER_OPT_WARN_ALL_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("/W4" COMPILER_OPT_W4_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-Wno-sign-compare" COMPILER_OPT_NO_WARN_SIGN_COMPARE_SUPPORTED)

set(SCAMP_ALL_BUILD_FLAGS "")
set(SCAMP_OPTIMIZED_BUILD_FLAGS "")

if (COMPILER_OPT_PIC_SUPPORTED)
  set(SCAMP_ALL_BUILD_FLAGS "${SCAMP_ALL_BUILD_FLAGS} -fPIC")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# GCC/Clang arch-specific optimizations
if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED AND NOT SCAMP_ENABLE_BINARY_DISTRIBUTION)
  set(SCAMP_OPTIMIZED_BUILD_FLAGS "${SCAMP_OPTIMIZED_BUILD_FLAGS} -march=native")
endif()

# ICC arch-specific optimizations
if (COMPILER_OPT_XHOST_SUPPORTED AND NOT SCAMP_ENABLE_BINARY_DISTRIBUTION)
  set(SCAMP_OPTIMIZED_BUILD_FLAGS "${SCAMP_OPTIMIZED_BUILD_FLAGS} -xHost")
endif()

if (COMPILER_OPT_QXHOST_SUPPORTED AND NOT SCAMP_ENABLE_BINARY_DISTRIBUTION)
  set(SCAMP_OPTIMIZED_BUILD_FLAGS "${SCAMP_OPTIMIZED_BUILD_FLAGS} /QxHost")
endif()

if (COMPILER_OPT_O3_SUPPORTED)
  set(SCAMP_OPTIMIZED_BUILD_FLAGS "${SCAMP_OPTIMIZED_BUILD_FLAGS} -O3")
endif()
  
if (COMPILER_OPT_UNROLL_LOOPS_SUPPORTED)
  set(SCAMP_OPTIMIZED_BUILD_FLAGS "${SCAMP_OPTIMIZED_BUILD_FLAGS} -funroll-loops")
endif()

if (COMPILER_OPT_FPCONTRACT_FAST_SUPPORTED)
  set(SCAMP_OPTIMIZED_BUILD_FLAGS "${SCAMP_OPTIMIZED_BUILD_FLAGS} -ffp-contract=fast")
endif()

if (COMPILER_OPT_FPCONTRACT_SUPPORTED)
  set(SCAMP_OPTIMIZED_BUILD_FLAGS "${SCAMP_OPTIMIZED_BUILD_FLAGS} /fp:contract")
endif()

if (COMPILER_OPT_FPPRECISE_SUPPORTED)
  set(SCAMP_OPTIMIZED_BUILD_FLAGS "${SCAMP_OPTIMIZED_BUILD_FLAGS} /fp:precise")
endif()

if (COMPILER_OPT_WARN_ALL_SUPPORTED AND NOT MSVC)
  set(SCAMP_ALL_BUILD_FLAGS "${SCAMP_ALL_BUILD_FLAGS} -Wall")
endif()

if (COMPILER_OPT_W4_SUPPORTED AND MSVC)
  set(SCAMP_ALL_BUILD_FLAGS "${SCAMP_ALL_BUILD_FLAGS} /W4")
endif()

if (COMPILER_OPT_NO_WARN_SIGN_COMPARE_SUPPORTED)
  set(SCAMP_ALL_BUILD_FLAGS "${SCAMP_ALL_BUILD_FLAGS} -Wno-sign-compare")
endif()

CHECK_CXX_COMPILER_FLAG("-fsanitize=address" COMPILER_OPT_SANITIZE_ADDRESS_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-fno-omit-frame-pointer" COMPILER_OPT_NO_OMIT_FP_SUPPORTED)

if (COMPILER_OPT_SANITIZE_ADDRESS_SUPPORTED)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
endif()

if (COMPILER_OPT_NO_OMIT_FP_SUPPORTED)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${SCAMP_ALL_BUILD_FLAGS} ${SCAMP_OPTIMIZED_BUILD_FLAGS}")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} ${SCAMP_ALL_BUILD_FLAGS} ${SCAMP_OPTIMIZED_BUILD_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${SCAMP_ALL_BUILD_FLAGS} ${SCAMP_OPTIMIZED_BUILD_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SCAMP_ALL_BUILD_FLAGS}")

# Only include Eigen code licensed under MPL2
add_definitions("-DEIGEN_MPL2_ONLY")

# ----------------------------------------------------------------------------------------
#                              ===== Build targets =====

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/")

# Add common targets.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/common)

if (BUILD_CLIENT_SERVER)
  # Build scamp client and scamp server
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/distributed)
endif()

# Add core functional libraries.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/core)

# Python module
if (BUILD_PYTHON_MODULE)
  message(STATUS "Configure Python Module")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/python)
  # If we are building pyscamp from setup.py we only care about the python module.
  if (BUILDING_PYSCAMP)
    return()
  endif()
endif()

if (BUILD_BENCHMARKS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark)
endif()

if(NOT TARGET gflags)
  add_subdirectory(third_party/gflags)
  include_directories(third_party/gflags/include/)
else()
  # We already produced gflags target from building the SCAMP client and server
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc/third_party/gflags/include/)
endif()

# C++/CUDA executable
add_executable(SCAMP ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
if (CMAKE_CUDA_COMPILER)
  target_compile_definitions(SCAMP PUBLIC -D_HAS_CUDA_)
  target_link_libraries(SCAMP ${CUDA_cudart_LIBRARY} gflags common scamp_interface scamp_utils)
else()
  target_link_libraries(SCAMP gflags common scamp_interface scamp_utils)
endif() 

# clang-format target
function(prepend var prefix)
  set(listVar "")

  foreach(f ${ARGN})
    list(APPEND listVar "${prefix}/${f}")
  endforeach()

  set(${var} "${listVar}" PARENT_SCOPE)
endfunction()

if(CLANG_FORMAT_EXE)
  prepend(FILES_TO_FORMAT ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_FILES})

  add_custom_target(
    clang-format
    COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${FILES_TO_FORMAT}
  )
endif()
