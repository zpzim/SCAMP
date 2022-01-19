cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include(FetchContent)
FetchContent_Declare(benchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           "v1.6.1"
)

set(BENCHMARK_DOWNLOAD_DEPENDENCIES 1)

FetchContent_MakeAvailable(benchmark)


add_executable(scamp_benchmarks "benchmarks.cpp")
target_link_libraries(scamp_benchmarks scamp_interface benchmark::benchmark)

