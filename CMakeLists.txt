# cmake file to build the project and tests
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# ----------------------------------------------------------------------------------------
#                              ===== Project Setup =====
project(SCAMP LANGUAGES CXX CUDA)
set(VERSION_MAJOR 1)
set(VERSION_MINOR 1)

find_program(
  CLANG_TIDY_EXE
  NAMES clang-tidy-6.0 clang-tidy-5.0 clang-tidy
  DOC "Path to clang-tidy executable (v5+)"
  )
if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found.")
else()
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-checks=*,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-fuchsia-default-arguments,-hicpp-vararg,-cppcoreguidelines-pro-type-vararg" "-warnings-as-errors=*")
endif()
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


add_subdirectory(proto)


set(
  PROJECT_SOURCE_FILES
  src/common.h
  src/kernels.h
  src/utils.h
  src/tile.h
  src/SCAMP.h
  src/fft_helper.h
  src/fft_kernels.h
  src/kernels.cu
  src/utils.cu
  src/tile.cpp
  src/SCAMP.cpp
  src/fft_helper.cpp
  src/fft_kernels.cu
  src/main.cpp
  src/common.cpp
  )

# ----------------------------------------------------------------------------------------
#                         ===== Compiler Configuration =====
add_definitions("--std=c++14")
message(STATUS "CUDA Root Directory is ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include" "proto")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_COMPILATION_FLAGS "-lineinfo --expt-relaxed-constexpr -use_fast_math -O3")
set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_70,code=sm_70")
set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_60")
#set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_50,code=sm_50")
#set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_37,code=sm_37")
#set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_35,code=sm_35")
#set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_30,code=sm_30")

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

add_library(common src/common.cpp)
target_link_libraries(common messages -lcudart_static)

add_library(fft_kernels src/fft_kernels.cu)
add_library(fft src/fft_helper.cpp)
target_link_libraries(fft -lcufft fft_kernels common)

add_library(utils src/utils.cu)
target_link_libraries(utils messages common)

add_library(kernels src/kernels.cu)
target_link_libraries(kernels messages common)

add_library(tile src/tile.cpp)
target_link_libraries(tile messages common kernels fft)
add_library(scamp src/SCAMP.cpp)
target_link_libraries(scamp messages utils tile common fft)


set_target_properties(kernels PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    COMPILE_FLAGS "${CUDA_COMPILATION_FLAGS} ${CUDA_GENCODE_FLAGS}"
)
set_target_properties(utils PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    COMPILE_FLAGS "${CUDA_COMPILATION_FLAGS} ${CUDA_GENCODE_FLAGS}"
)

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
    tile PROPERTIES
    CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
  )
  set_target_properties(
    fft PROPERTIES
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
target_link_libraries(SCAMP -lcudart_static messages common scamp)
