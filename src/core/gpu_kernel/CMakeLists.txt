include(SCAMPMacros)

# This subdirectory is only built when CUDA is available. It collects all
# GPU-specific kernel code into a single 'gpu_kernels' library plus a small
# 'qt_kernels' helper library for the cuFFT-based sliding dot product.

add_library(qt_kernels ${CMAKE_CURRENT_SOURCE_DIR}/qt_kernels.cu)

add_library(gpu_utils ${CMAKE_CURRENT_SOURCE_DIR}/kernel_gpu_utils.cu)
target_link_libraries(gpu_utils kernel_common)

add_library(gpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu)
target_link_libraries(gpu_kernels gpu_utils kernel_common common)

mark_cuda_if_available()
mark_clang_tidy()
