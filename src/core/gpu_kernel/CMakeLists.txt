include(SCAMPMacros)

if (NOT CMAKE_CUDA_COMPILER)
  message(WARNING "Not building GPU kernels because no CUDA compiler was available.")
  return()
endif()

# Compute the initial sliding dot products using the GPU + cufft.
add_library(qt_kernels ${CMAKE_CURRENT_SOURCE_DIR}/qt_kernels.cu)
target_link_libraries(qt_helper CUDA::cufft qt_kernels)

add_library(gpu_utils ${CMAKE_CURRENT_SOURCE_DIR}/kernel_gpu_utils.cu)
target_link_libraries(gpu_utils kernel_common)
add_library(gpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu)
target_link_libraries(gpu_kernels gpu_utils kernel_common common Eigen3::Eigen)

mark_cuda_if_available()
mark_clang_tidy()
