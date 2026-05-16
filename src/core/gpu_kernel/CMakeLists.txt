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

# Embed data/autotune_cache.txt into the binary as a C++ string constant.
# The runtime lookup consults this built-in cache after user overrides; it is
# the only way conda-forge / pip wheel users get GPU-specific tuning, since
# they cannot recompile. To update the contents for a new device, run the
# autotuner locally and merge the result into data/autotune_cache.txt, then
# rebuild.
file(READ "${CMAKE_SOURCE_DIR}/data/autotune_cache.txt"
     SCAMP_BUILTIN_AUTOTUNE_CONTENT)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/builtin_autotune_cache.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/builtin_autotune_cache.cpp"
    @ONLY)

add_library(gpu_utils
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_gpu_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_props.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autotune_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autotune.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/builtin_autotune_cache.cpp
)
target_include_directories(gpu_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gpu_utils kernel_common common CUDA::cudart)

add_library(gpu_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_1nn.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_1nn_index.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sum_thresh.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_matrix_summary.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_approx_all_neighbors.cu
)
# Eigen3::Eigen is defined by src/core/cpu_kernel/CMakeLists.txt, which the
# parent src/core/CMakeLists.txt processes before this directory. We use
# Eigen's fixed-size arrays in the device-side compute templates for cleaner
# expression syntax than hand-unrolled C arrays; the headers are
# device-callable when --expt-relaxed-constexpr is on (set at top level).
target_link_libraries(gpu_kernels gpu_utils kernel_common common Eigen3::Eigen)

mark_cuda_if_available()
mark_clang_tidy()
