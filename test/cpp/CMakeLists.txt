# C++ unit tests for SCAMP. Currently CPU-only -- the tests link against
# gpu_utils for symbols (AutotuneCache, LookupKernelConfigForDeviceKey)
# but don't actually launch any CUDA at runtime, so they run on machines
# without a GPU even though they require a CUDA toolchain to *build*.
#
# Each test executable is registered with ctest via add_test() so the
# usual `ctest --output-on-failure` workflow surfaces failures.

if (NOT CMAKE_CUDA_COMPILER)
  # The current test suite covers GPU-side machinery (autotune cache).
  # If there's no CUDA, gpu_utils doesn't exist and there's nothing
  # meaningful to link against. Future CPU-only tests would land in a
  # sibling block here.
  message(STATUS "SCAMP unit tests skipped (CUDA not available).")
  return()
endif()

add_executable(test_autotune_cache test_autotune_cache.cpp)
target_include_directories(test_autotune_cache
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
# gpu_utils provides AutotuneCache, LookupKernelConfigForDeviceKey, the
# enum-name helpers, and the IsSupportedKernelConfig validator. We link
# the CUDA runtime transitively but never call into it; the test exits
# without touching any device.
target_link_libraries(test_autotune_cache PRIVATE gpu_utils common)
add_test(NAME test_autotune_cache COMMAND test_autotune_cache)
