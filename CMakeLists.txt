# cmake file to build the project and tests
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
include(CheckLanguage)
# ----------------------------------------------------------------------------------------
#                              ===== Project Setup =====
project(SCAMP LANGUAGES CXX)
set(VERSION_MAJOR 1)
set(VERSION_MINOR 1)

if(FORCE_NO_CUDA)
  unset(CMAKE_CUDA_COMPILER)
else()
  # Use cuda if available
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
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
  )

# ----------------------------------------------------------------------------------------
#                         ===== Compiler Configuration =====

add_definitions("-std=c++14")

# CUDA_CONFIG
if (CMAKE_CUDA_COMPILER)
  add_definitions("-D_HAS_CUDA_")
  find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_COMPILATION_FLAGS "-lineinfo --expt-relaxed-constexpr -use_fast_math -O3")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_70")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_61,code=sm_61")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_60")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_50,code=sm_50")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_37,code=sm_37")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_35,code=sm_35")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_30,code=sm_30")
endif()

 
# optimize in release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# build debug with sanitizer
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")

if(${COMPILER_ID} MATCHES "GNU") # gcc specific flags
    add_definitions("-Wextra -Wpedantic")
elseif(${COMPILER_ID} MATCHES "clang") # clang specific flags
    add_definitions("-Weverything")
endif()

# ----------------------------------------------------------------------------------------
#                              ===== Build targets =====

add_subdirectory(gflags)
add_subdirectory(proto)
include_directories("proto")
add_library(common src/common.cpp)

if (CMAKE_CUDA_COMPILER) 
  target_link_libraries(common messages -lcudart_static)
else()
  target_link_libraries(common messages)
endif()

add_library(qt_helper src/qt_helper.cpp)

if (CMAKE_CUDA_COMPILER)
  add_library(qt_kernels src/qt_kernels.cu)
  target_link_libraries(qt_helper -lcufft qt_kernels common)
else()
  target_link_libraries(qt_helper common)
endif()


if (CMAKE_CUDA_COMPILER)
  add_library(gpu_stats src/gpu_stats.cu)
  target_link_libraries(gpu_stats messages common)
  add_library(gpu_kernels src/kernels.cu)
  target_link_libraries(gpu_kernels messages common)
else()
  add_library(cpu_stats src/cpu_stats.cpp)
  target_link_libraries(cpu_stats messages common)
endif()

add_library(cpu_kernels src/cpu_kernels.cpp)
target_link_libraries(cpu_kernels messages common)

add_library(tile src/tile.cpp)


if (CMAKE_CUDA_COMPILER)
  target_link_libraries(tile messages common gpu_kernels cpu_kernels qt_helper)
else()
  target_link_libraries(tile messages common cpu_kernels qt_helper)
endif()

add_library(scamp src/SCAMP.cpp)

if (CMAKE_CUDA_COMPILER)
  target_link_libraries(scamp -lpthread tile messages gpu_stats common qt_helper)
else()
  target_link_libraries(scamp -lpthread tile messages cpu_stats common qt_helper)
endif() 

if (CMAKE_CUDA_COMPILER)
set_target_properties(gpu_kernels PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    COMPILE_FLAGS "${CUDA_COMPILATION_FLAGS} ${CUDA_GENCODE_FLAGS}"
)
set_target_properties(gpu_stats PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    COMPILE_FLAGS "${CUDA_COMPILATION_FLAGS} ${CUDA_GENCODE_FLAGS}"
)
endif()

if(CLANG_TIDY_EXE)
  set_target_properties(
    common PROPERTIES
    CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
  )
  set_target_properties(
    scamp PROPERTIES
    CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
  )
  set_target_properties(
    qt_helper PROPERTIES
    CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
  )
  set_target_properties(
    tile PROPERTIES
    CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
  )
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

add_executable(SCAMP src/main.cpp)
if (CMAKE_CUDA_COMPILER)
  target_link_libraries(SCAMP -lcudart_static gflags messages common scamp)
else()
  target_link_libraries(SCAMP gflags messages common scamp)
endif() 
