include(SCAMPMacros)

# Build target for computing Sliding Dot Product. qt_helper has both CPU and
# GPU code paths (gated on _HAS_CUDA_); the cuFFT-based GPU kernels it calls
# into live in src/core/gpu_kernel/.
add_library(qt_helper ${CMAKE_CURRENT_SOURCE_DIR}/qt_helper.cpp)

# Build shared SCAMP kernel-info target (used by both CPU and GPU paths).
add_library(kernel_common ${CMAKE_CURRENT_SOURCE_DIR}/kernel_common.cpp)
if(USE_HIP)
  target_link_libraries(kernel_common hip::host)
endif()

# Defines 'cpu_kernels' target.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpu_kernel)

if(USE_HIP OR CMAKE_CUDA_COMPILER)
  # Defines 'gpu_kernels', 'gpu_utils', and 'qt_kernels' targets.
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu_kernel)
  if(USE_HIP)
    target_link_libraries(qt_helper hip::hipfft qt_kernels common)
  else()
    target_link_libraries(qt_helper CUDA::cufft qt_kernels common)
  endif()
else()
  target_link_libraries(qt_helper common)
endif()

add_library(cpu_stats ${CMAKE_CURRENT_SOURCE_DIR}/cpu_stats.cpp)
target_link_libraries(cpu_stats common)

add_library(tile ${CMAKE_CURRENT_SOURCE_DIR}/tile.cpp)

if(USE_HIP OR CMAKE_CUDA_COMPILER)
  target_link_libraries(tile common gpu_kernels cpu_kernels profile qt_helper)
else()
  target_link_libraries(tile common cpu_kernels profile qt_helper)
endif()

add_library(scamp_op ${CMAKE_CURRENT_SOURCE_DIR}/SCAMP.cpp)
target_link_libraries(scamp_op Threads::Threads tile cpu_stats common profile qt_helper)

mark_cuda_if_available()
mark_clang_tidy()
