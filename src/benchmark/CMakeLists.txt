cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include(FetchContent)
FetchContent_Declare(benchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           "v1.6.1"
)

set(BENCHMARK_DOWNLOAD_DEPENDENCIES 1)

FetchContent_MakeAvailable(benchmark)

add_library(scamp_benchmarks_common "common.cpp")
target_link_libraries(scamp_benchmarks_common scamp_interface benchmark::benchmark)

add_executable(scamp_cpu_benchmarks "cpu_benchmarks.cpp")
target_link_libraries(scamp_cpu_benchmarks scamp_benchmarks_common benchmark::benchmark)

if (CMAKE_CUDA_COMPILER)
  add_executable(scamp_gpu_benchmarks "gpu_benchmarks.cpp")
  target_link_libraries(scamp_gpu_benchmarks scamp_benchmarks_common benchmark::benchmark)
endif()


