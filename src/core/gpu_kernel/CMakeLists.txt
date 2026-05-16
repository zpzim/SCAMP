include(SCAMPMacros)

# This subdirectory is only built when CUDA is available. It collects all
# GPU-specific kernel code into a single 'gpu_kernels' library plus a small
# 'qt_kernels' helper library for the cuFFT-based sliding dot product.
#
# The five do_tile<...> template instantiations (one per profile type) live in
# their own .cu files so they compile in parallel; kernels.cu only contains
# the slim runtime dispatcher and entry points. Add a new kernel_<name>.cu
# here when adding a profile type.

add_library(qt_kernels ${CMAKE_CURRENT_SOURCE_DIR}/qt_kernels.cu)

add_library(gpu_utils ${CMAKE_CURRENT_SOURCE_DIR}/kernel_gpu_utils.cu)
target_link_libraries(gpu_utils kernel_common)

add_library(gpu_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_1nn.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_1nn_index.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sum_thresh.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_matrix_summary.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_approx_all_neighbors.cu
)
target_link_libraries(gpu_kernels gpu_utils kernel_common common)

mark_cuda_if_available()
mark_clang_tidy()
