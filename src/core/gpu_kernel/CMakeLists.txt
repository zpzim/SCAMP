include(SCAMPMacros)

# This subdirectory is only built when CUDA is available. It collects all
# GPU-specific kernel code into a single 'gpu_kernels' library plus a small
# 'qt_kernels' helper library for the cuFFT-based sliding dot product.
#
# Layout for parallel builds:
#   kernels.cu               -- top-level dispatcher (profile_type -> LaunchKernel_<X>)
#   kernel_<profile>.cu      -- per-profile dispatcher (cfg switch via
#                               SCAMP_VARIANT_DISPATCH macro -> LaunchVariant_<X>_vN)
#   kernel_<profile>_v<N>.cu -- auto-generated (configure_file) from
#                               kernel_variant.cu.in; each instantiates the heavy
#                               do_tile<...> kernel for exactly one variant
#                               geometry, across the 2 precisions x 3 row/col modes.
#
# 5 profiles x 6 variants = 30 generated TUs. Splitting this way lets the
# build dispatch variant compilation to make/ninja's -jN. Adding a profile:
# add an entry to SCAMP_KERNEL_PROFILES below + create a kernel_<profile>.cu
# dispatcher + add the SCAMP_DECL_VARIANTS_FOR_PROFILE block in
# kernels_variants.h. Adding a variant: add an entry to SCAMP_VARIANT_TUPLES
# below + add an entry to kKernelVariants (kernel_config.cpp) + add the
# matching dispatch case in SCAMP_VARIANT_DISPATCH (kernels_variants.h).

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
# gpu_utils needs Eigen too: kernel_gpu_utils.h now declares Eigen::Map
# fields inside SCAMPSmem and Eigen::Array inside SCAMPThreadInfo.
target_link_libraries(gpu_utils kernel_common common CUDA::cudart Eigen3::Eigen)

# Per-profile metadata: name + PROFILE_OUTPUT_TYPE + PROFILE_DATA_TYPE +
# DISTANCE_TYPE. Used by the configure_file foreach below to fill in the
# concrete type template arguments for each generated kernel_<P>_v<N>.cu.
# Keep in sync with kernels_variants.h's SCAMP_DECL_VARIANTS_FOR_PROFILE
# instantiations and with the per-profile LaunchKernel_<X> signatures.
set(SCAMP_KERNEL_PROFILES
    "1NN|float|float|float"
    "1NN_INDEX|uint64_t|uint64_t|float"
    "SUM_THRESH|double|double|double"
    "MATRIX_SUMMARY|float|uint64_t|float"
    "APPROX_ALL_NEIGHBORS|SCAMPmatch|uint64_t|float"
)

# Variant geometry tuples: index|bps|dpt|ur|our|kti. MUST stay in sync with
# kKernelVariants in kernel_config.cpp and the SCAMP_VARIANT_DISPATCH cases
# in kernels_variants.h. Adding a row here without updating the other two
# generates a .cu that compiles but is never callable (and one whose
# IsSupportedKernelConfig returns false).
set(SCAMP_VARIANT_TUPLES
    # Bake-off: sliding-window variants v0/v1/v3/v4 enabled alongside v6
    # (shfl) so the autotune cache can route to any of them for perf
    # comparison. v2/v5 still skipped (v2 = v1 with deeper unroll, v5 is
    # bps=1 niche).
    "0|2|2|2|16|16"
    "1|2|4|2|4|50"
    "2|2|4|4|4|50"
    "3|4|2|2|8|16"
    "4|2|2|2|8|32"
    "5|1|4|4|16|16"
    # ur==0 marks the design-A "shfl" variant. The foreach below routes
    # ur==0 tuples through kernel_variant_shfl.cu.in instead of the
    # sliding-window template.
    "6|8|4|0|8|8"
    "7|8|4|0|8|16"
    "8|4|8|0|8|32"
)

set(SCAMP_GENERATED_VARIANT_SOURCES "")
foreach(profile_def ${SCAMP_KERNEL_PROFILES})
  string(REPLACE "|" ";" profile_fields "${profile_def}")
  list(GET profile_fields 0 VARIANT_PROFILE)
  list(GET profile_fields 1 VARIANT_PROFILE_OUTPUT_TYPE)
  list(GET profile_fields 2 VARIANT_PROFILE_DATA_TYPE)
  list(GET profile_fields 3 VARIANT_DISTANCE_TYPE)
  foreach(variant_tuple ${SCAMP_VARIANT_TUPLES})
    string(REPLACE "|" ";" variant_fields "${variant_tuple}")
    list(GET variant_fields 0 VARIANT_INDEX)
    list(GET variant_fields 1 VARIANT_BPS)
    list(GET variant_fields 2 VARIANT_DPT)
    list(GET variant_fields 3 VARIANT_UR)
    list(GET variant_fields 4 VARIANT_OUR)
    list(GET variant_fields 5 VARIANT_KTI)
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/kernel_${VARIANT_PROFILE}_v${VARIANT_INDEX}.cu")
    if(VARIANT_UR EQUAL 0)
      set(_template "${CMAKE_CURRENT_SOURCE_DIR}/kernel_variant_shfl.cu.in")
    else()
      set(_template "${CMAKE_CURRENT_SOURCE_DIR}/kernel_variant.cu.in")
    endif()
    configure_file(
      "${_template}"
      "${_out}"
      @ONLY)
    list(APPEND SCAMP_GENERATED_VARIANT_SOURCES "${_out}")
  endforeach()
endforeach()

add_library(gpu_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_1nn.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_1nn_index.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sum_thresh.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_matrix_summary.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_approx_all_neighbors.cu
    ${SCAMP_GENERATED_VARIANT_SOURCES}
)
target_include_directories(gpu_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Eigen3::Eigen is defined by src/core/cpu_kernel/CMakeLists.txt, which the
# parent src/core/CMakeLists.txt processes before this directory. We use
# Eigen's fixed-size arrays in the device-side compute templates for cleaner
# expression syntax than hand-unrolled C arrays; the headers are
# device-callable when --expt-relaxed-constexpr is on (set at top level).
target_link_libraries(gpu_kernels gpu_utils kernel_common common Eigen3::Eigen)

# autotune_bench: the autotune benchmark loop. Lives in its own library so
# the dep on scamp_op (needed to invoke do_SCAMP per-trial) doesn't have to
# go into gpu_utils, which would create a cycle (scamp_op transitively
# depends on gpu_kernels -> gpu_utils). Only callers that want the
# benchmarked autotune (currently main.cpp + pyscamp) link this. scamp_op
# is defined later in src/core/CMakeLists.txt; modern CMake resolves the
# forward target reference at generate time.
add_library(autotune_bench ${CMAKE_CURRENT_SOURCE_DIR}/autotune_bench.cpp)
target_include_directories(autotune_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(autotune_bench gpu_utils scamp_op scamp_args common)

mark_cuda_if_available()
mark_clang_tidy()
