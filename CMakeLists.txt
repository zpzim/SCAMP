# cmake file to build the project and tests
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
include(CheckLanguage)
include(CheckCXXCompilerFlag)
# ----------------------------------------------------------------------------------------
#                              ===== Project Setup =====
project(SCAMPmain LANGUAGES CXX)
set(VERSION_MAJOR 1)
set(VERSION_MINOR 1)

set(CUDA_MINIMUM_VERSION "9.0")

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
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
    message(STATUS "CUDA version ${CUDA_VERSION} is less than the mininmum required version ${CUDA_MINIMUM_VERSION} defaulting to no-cuda build")
    unset(CMAKE_CUDA_COMPILER)
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
endif()

# Use clang tidy if available
find_program(
  CLANG_TIDY_EXE
  NAMES clang-tidy-6.0 clang-tidy-5.0 clang-tidy
  DOC "Path to clang-tidy executable (v5+)"
  )
if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found.")
else()
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-checks=*,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-fuchsia-default-arguments,-hicpp-vararg,-cppcoreguidelines-pro-type-vararg")
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
  src/*.h
  src/*.cpp
  src/*.cu
  kubernetes/*.cc
  kubernetes/*.cpp
  kubernetes/*.h
  python/*.cpp
  )

# ----------------------------------------------------------------------------------------
#                         ===== Compiler Configuration =====

set(CMAKE_CXX_STANDARD 11)

# CUDA_CONFIG
if (CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_STANDARD 11)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC -lineinfo")
  if (CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_75,code=sm_75")
  endif()
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_70")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_61,code=sm_61")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_60")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_50,code=sm_50")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_37,code=sm_37")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_35,code=sm_35")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_30,code=sm_30")
endif()

CHECK_CXX_COMPILER_FLAG("-fPIC" COMPILER_OPT_PIC_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-O3" COMPILER_OPT_O3_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-funroll-loops" COMPILER_OPT_UNROLL_LOOPS_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-ffp-contract=fast" COMPILER_OPT_FPCONTRACT_FAST_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-Wall" COMPILER_OPT_WARN_ALL_SUPPORTED)

if (COMPILER_OPT_PIC_SUPPORTED)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
endif()

if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

if (COMPILER_OPT_O3_SUPPORTED)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()
  
if (COMPILER_OPT_UNROLL_LOOPS_SUPPORTED)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
endif()

if (COMPILER_OPT_FPCONTRACT_FAST_SUPPORTED)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=fast")
endif()

if (COMPILER_OPT_WARN_ALL_SUPPORTED)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
endif()

CHECK_CXX_COMPILER_FLAG("-fsanitize=address" COMPILER_OPT_SANITIZE_ADDRESS_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-fno-omit-frame-pointer" COMPILER_OPT_NO_OMIT_FP_SUPPORTED)

if (COMPILER_OPT_SANITIZE_ADDRESS_SUPPORTED)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
endif()

if (COMPILER_OPT_NO_OMIT_FP_SUPPORTED)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
endif()


# ----------------------------------------------------------------------------------------
#                              ===== Build targets =====

if (BUILD_PYTHON_MODULE)
  find_package(PythonLibs)
  add_subdirectory(pybind11)
endif()

add_library(common ${CMAKE_CURRENT_SOURCE_DIR}/src/common.cpp)

if (CMAKE_CUDA_COMPILER) 
  target_link_libraries(common ${CUDA_cudart_LIBRARY})
endif()

add_library(qt_helper ${CMAKE_CURRENT_SOURCE_DIR}/src/qt_helper.cpp)

if (CMAKE_CUDA_COMPILER)
  add_library(qt_kernels ${CMAKE_CURRENT_SOURCE_DIR}/src/qt_kernels.cu)
  target_link_libraries(qt_helper ${CUDA_CUFFT_LIBRARIES} qt_kernels common)
else()
  target_link_libraries(qt_helper common)
endif()

add_library(kernel_common ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_common.cpp)

if (CMAKE_CUDA_COMPILER)
  add_library(gpu_utils ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_gpu_utils.cu)
  target_link_libraries(gpu_utils kernel_common ${CUDA_cudart_LIBRARY})
  add_library(gpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cu)
  target_link_libraries(gpu_kernels gpu_utils kernel_common common)
endif()

add_library(cpu_stats ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_stats.cpp)
target_link_libraries(cpu_stats common)

add_library(cpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_kernels.cpp)
target_link_libraries(cpu_kernels kernel_common common)

add_library(tile ${CMAKE_CURRENT_SOURCE_DIR}/src/tile.cpp)


if (CMAKE_CUDA_COMPILER)
  target_link_libraries(tile common gpu_kernels cpu_kernels qt_helper)
else()
  target_link_libraries(tile common cpu_kernels qt_helper)
endif()

add_library(scamp_utils ${CMAKE_CURRENT_SOURCE_DIR}/src/scamp_utils.cpp)

add_library(scamp_op ${CMAKE_CURRENT_SOURCE_DIR}/src/SCAMP.cpp)
target_link_libraries(scamp_op Threads::Threads tile cpu_stats common qt_helper)


if (CMAKE_CUDA_COMPILER)
  set_target_properties(gpu_kernels PROPERTIES
    COMPILE_FLAGS "${COMPILE_FLAGS} ${CUDA_GENCODE_FLAGS}"
  )
endif()

# C++/CUDA executable
add_executable(SCAMP ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# Python module
if (BUILD_PYTHON_MODULE)
  add_library(pyscamp MODULE python/SCAMP_python.cpp)
  set_target_properties(pyscamp PROPERTIES 
    INTERPROCEDURAL_OPTIMIZATION TRUE
    CXX_VISIBILITY_PRESET "hidden"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )
  target_link_libraries(pyscamp PRIVATE scamp_op scamp_utils common)
  target_link_libraries(pyscamp PRIVATE pybind11::module)
endif()

if (CMAKE_CUDA_COMPILER)
  target_compile_definitions(common PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(qt_helper PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(qt_kernels PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(kernel_common PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(gpu_utils PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(gpu_kernels PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(cpu_stats PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(cpu_kernels PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(tile PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(scamp_utils PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(scamp_op PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(gpu_kernels PUBLIC -D_HAS_CUDA_)
  target_compile_definitions(SCAMP PUBLIC -D_HAS_CUDA_)
  if (BUILD_PYTHON_MODULE)
    target_compile_definitions(pyscamp PUBLIC -D_HAS_CUDA_)
  endif()
endif()



if (BUILD_CLIENT_SERVER)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/grpc EXCLUDE_FROM_ALL)
  message(STATUS "Using gRPC via add_subdirectory.")
  
  # After using add_subdirectory, we can now use the grpc targets directly from
  # this build.
  set(_PROTOBUF_LIBPROTOBUF libprotobuf)
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
  set(_GRPC_GRPCPP_UNSECURE grpc++_unsecure)
  set(_GRPC_GRPCPP grpc++)
  set(_GRPC_GRPC grpc)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

  # Add include directories we will use later
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/grpc/include/) 
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/grpc/third_party/gflags/include/)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/grpc/third_party/protobuf/src/)

  # Build scamp client and scamp server
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kubernetes)

endif()

if(NOT TARGET gflags)
  add_subdirectory(gflags)
  include_directories(gflags/include/)
endif()

if (CMAKE_CUDA_COMPILER)
  target_link_libraries(SCAMP ${CUDA_cudart_LIBRARY} gflags common scamp_op scamp_utils)
else()
  target_link_libraries(SCAMP gflags common scamp_op scamp_utils)
endif() 

# Add clang tidy rules
if(CLANG_TIDY_EXE)
  set(MAIN_LIBS common scamp_op qt_helper tile scamp_utils cpu_kernels cpu_stats)
  foreach(lib ${MAIN_LIBS})
    set_target_properties(
      "${lib}" PROPERTIES
      CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
  endforeach(lib)
  if (CMAKE_CUDA_COMPILER)
    set_target_properties(
      gpu_kernels PROPERTIES
      CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
  endif()

endif()

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
