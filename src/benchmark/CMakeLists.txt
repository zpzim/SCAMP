cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include(FetchContent)
FetchContent_Declare(benchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           "v1.9.5"
)

# Disable benchmark's own test suite so it does not try to download GoogleTest.
# GoogleTest's old cmake_minimum_required is rejected by CMake 4.x and the
# download runs in a subprocess that does not inherit CMAKE_POLICY_VERSION_MINIMUM.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

# Disable benchmark's -Werror. SCAMP's global CMAKE_CXX_FLAGS_RELEASE adds
# Linux-style flags like -O3 which clang-cl warns about as unused arguments;
# combined with benchmark's -Werror this turns warnings into hard errors.
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

# On MSVC (cl), benchmark's regex backend detection fails because its test
# compiles a snippet that gates on __cplusplus, which MSVC reports as 199711L
# unless /Zc:__cplusplus is set. Pre-cache the result; std::regex is
# available on every supported MSVC toolchain.
if(MSVC)
  set(HAVE_STD_REGEX ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(benchmark)

add_library(scamp_benchmarks_common "common.cpp")
target_link_libraries(scamp_benchmarks_common scamp_interface benchmark::benchmark)

add_executable(scamp_cpu_benchmarks "cpu_benchmarks.cpp")
target_link_libraries(scamp_cpu_benchmarks scamp_benchmarks_common benchmark::benchmark)

if (CMAKE_CUDA_COMPILER)
  add_executable(scamp_gpu_benchmarks "gpu_benchmarks.cpp")
  target_link_libraries(scamp_gpu_benchmarks scamp_benchmarks_common benchmark::benchmark)
endif()


