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
# One TU per (profile, variant) lets make/ninja dispatch variant
# compilation to -jN.
#
# Adding a profile: add an entry to SCAMP_KERNEL_PROFILES below + create
# a kernel_<profile>.cu dispatcher.
# Adding a variant: add an entry to SCAMP_VARIANT_TUPLES below; the
# foreach in this file regenerates the dispatch table + kVariants[]
# entry + per-(profile, variant) .cu files automatically.

add_library(qt_kernels ${CMAKE_CURRENT_SOURCE_DIR}/qt_kernels.cu)

# Embed data/autotune_cache.txt into the binary as a C++ string constant.
# The runtime lookup consults this built-in cache after user overrides; it is
# the only way conda-forge / pip wheel users get GPU-specific tuning, since
# they cannot recompile. To update the contents for a new device, run the
# autotuner locally and merge the result into data/autotune_cache.txt, then
# rebuild.
#
# CMAKE_CONFIGURE_DEPENDS is REQUIRED here: file(READ ...) snapshots the file
# contents into a CMake variable at configure time and does NOT establish a
# dependency on the source file. Without this hint, editing
# data/autotune_cache.txt and re-running `cmake --build` will silently leave
# the old contents baked into the binary -- the user's new entries appear
# correct on disk but the runtime lookup misses them and falls through to
# the previous defaults. Listing the file here makes CMake re-run configure
# (and thus re-execute the file(READ) + configure_file) whenever it changes.
set_property(DIRECTORY "${CMAKE_SOURCE_DIR}"
             APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${CMAKE_SOURCE_DIR}/data/autotune_cache.txt")
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
target_include_directories(gpu_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                              ${CMAKE_CURRENT_BINARY_DIR})
# Eigen3::Eigen for the Eigen::Map / Eigen::Array fields in
# kernel_gpu_utils.h's SCAMPSmem / SCAMPThreadInfo.
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

# Variant geometry tuples -- the SINGLE source of truth for which kernel
# variants are enabled. Format: bps|dpt|ur|our|kti (no explicit index --
# variant N is the Nth entry in this list). The foreach below generates
# from this list:
#   * kernel_<profile>_v<N>.cu per (profile, variant) tuple via
#     kernel_variant.cu.in / kernel_variant_shfl.cu.in.
#   * kernel_variants_table.h with the kVariants[] init list consumed by
#     kernel_config.cpp.
#   * kernels_variants.h with the SCAMP_DECL_VARIANTS_FOR_PROFILE +
#     SCAMP_VARIANT_DISPATCH macros consumed by the per-profile .cu
#     dispatcher TUs.
#
# Adding or reordering a tuple here propagates through all three
# automatically; no other files need touching. Retiring a tuple removes
# its .cu, its forward decl, its dispatch case, and its kVariants entry
# in one step.
set(SCAMP_VARIANT_TUPLES
    # bps|dpt|ur|our|kti|default_blocksz_sp
    #
    # default_blocksz_sp is the SP cold-start blocksz the runtime uses when
    # the autotune cache misses for this variant. DP cold-start uses
    # default_blocksz_sp / 2 implicitly: DP uses 2x the registers per
    # thread, and halving threads/SM keeps the per-thread register budget
    # stable across precisions. The /2 happens in the launcher macros
    # (LaunchDoTileShflWithGeometry / LaunchDoTileWithGeometry) -- variant
    # authors do not declare it.
    #
    # The kernel still gets instantiated for every blocksz in {64,128,256,512}
    # (the autotuner sweeps that axis independently and the on-disk cache
    # can store any of them), so this default only matters on the very
    # first untuned run on a device.
    # Two sliding-window variants -- same big-tile geometry (DPT=4, UR=4,
    # OUR=16, KTI=16 -> tile=256) at two thread-density targets. bps=1
    # (target=512 thr/SM) wins DP for atomic-light profiles (1NN, AAN) where
    # the DP register footprint needs slack; bps=2 (target=1024 thr/SM) wins
    # SP variants of the same profiles where the SP footprint fits 2x more
    # blocks/SM. Higher bps (4) was tested and triggered a spill cliff on
    # sm_86; lower-occupancy and smaller-tile SW variants (DPT=2 baseline,
    # DPT=4 + 4x4 hand-unroll) were dropped after the consolidation autotune
    # showed they never won a target.
    "1|4|4|16|16|512"           # sliding-window, big tile, bps=1 (DP-friendly)
    "2|4|4|16|16|512"           # sliding-window, big tile, bps=2 (SP-friendly)
    # ur==0 marks the "shfl" (cov-shuffle) variant. The foreach below
    # routes ur==0 tuples through kernel_variant_shfl.cu.in instead of
    # the sliding-window template.
    #
    # Two shfl variants cover the precision split: a dpt=8 tile=256 variant
    # for SP (and DP 1NN, since SP can afford the larger per-thread state)
    # and a dpt=4 tile=128 variant for DP targets where halving the per-
    # thread state lets DP fit higher occupancy. Both default to bsz=128;
    # the autotuner sweeps the other blocksizes independently.
    "5|8|0|8|32|128"            # shfl, DPT=8, tile=256 (SP default + best overall geomean: 1.016x of per-target best)
    "8|4|0|8|16|128"            # shfl, DPT=4, tile=128 (DP champion: wins 1NN_INDEX DP + AAN DP via lower per-thread register footprint)
)

# Build the three @-substitution strings (kVariants init body, forward
# decl block, dispatch ladder body) by walking SCAMP_VARIANT_TUPLES.
# Variant N is the Nth entry in the list (foreach iteration counter);
# there are no explicit indices in the tuple strings.
list(LENGTH SCAMP_VARIANT_TUPLES SCAMP_NUM_VARIANTS)
set(_kvariants_init "")
set(_variant_decls "")
set(_variant_dispatch "")
set(_variant_idx 0)
set(SCAMP_GENERATED_VARIANT_SOURCES "")
foreach(variant_tuple ${SCAMP_VARIANT_TUPLES})
  string(REPLACE "|" ";" variant_fields "${variant_tuple}")
  list(GET variant_fields 0 VARIANT_BPS)
  list(GET variant_fields 1 VARIANT_DPT)
  list(GET variant_fields 2 VARIANT_UR)
  list(GET variant_fields 3 VARIANT_OUR)
  list(GET variant_fields 4 VARIANT_KTI)
  list(GET variant_fields 5 VARIANT_DEFAULT_BLOCKSZ_SP)
  set(VARIANT_INDEX ${_variant_idx})

  # 1) Per-variant kVariants[] initializer row. Newline-terminated so
  #    `configure_file` produces a multi-line array body. default_blocksz_dp
  #    is implicit (= default_blocksz_sp / 2), so the struct stores only
  #    default_blocksz_sp.
  string(APPEND _kvariants_init
    "    {${VARIANT_BPS}, ${VARIANT_DPT}, ${VARIANT_UR}, ${VARIANT_OUR}, ${VARIANT_KTI}, ${VARIANT_DEFAULT_BLOCKSZ_SP}},  // v${VARIANT_INDEX}\n")

  # 2) Per-variant forward decl, contributing to the body of the
  #    SCAMP_DECL_VARIANTS_FOR_PROFILE macro. Every line ends with \
  #    so the macro spans correctly. NB: in CMake quoted strings \\
  #    is the way to emit a single backslash literal.
  string(APPEND _variant_decls
    "  SCAMPError_t LaunchVariant_##PROFILE##_v${VARIANT_INDEX}(\\\n"
    "      SCAMPKernelInputArgs<double> args, OUTPUT_TYPE *profile_A,\\\n"
    "      OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type,\\\n"
    "      bool computing_rows, bool computing_cols, uint64_t blocksz,\\\n"
    "      uint64_t num_blocks, uint64_t smem, cudaStream_t s);\\\n")

  # 3) Per-variant dispatch if-block, contributing to the body of the
  #    SCAMP_VARIANT_DISPATCH macro.
  string(APPEND _variant_dispatch
    "    /* v${VARIANT_INDEX}: ${VARIANT_BPS},${VARIANT_DPT},${VARIANT_UR},${VARIANT_OUR},${VARIANT_KTI} */\\\n"
    "    if (cfg.blocks_per_sm == ${VARIANT_BPS} &&\\\n"
    "        cfg.diags_per_thread == ${VARIANT_DPT} &&\\\n"
    "        cfg.unrolled_rows == ${VARIANT_UR} &&\\\n"
    "        cfg.outer_unrolled_rows == ${VARIANT_OUR} &&\\\n"
    "        cfg.kernel_tile_iters == ${VARIANT_KTI}) {\\\n"
    "      return LaunchVariant_##PROFILE##_v${VARIANT_INDEX}(\\\n"
    "          args, profile_A, profile_B, fp_type, computing_rows,\\\n"
    "          computing_cols, cfg.blocksz, num_blocks, smem, s);\\\n"
    "    }\\\n")

  # 4) Per-(profile, variant) .cu generation: one TU that instantiates
  #    do_tile for this geometry across the 2 precisions x 3 row/col
  #    modes = 6 do_tile bodies per file.
  foreach(profile_def ${SCAMP_KERNEL_PROFILES})
    string(REPLACE "|" ";" profile_fields "${profile_def}")
    list(GET profile_fields 0 VARIANT_PROFILE)
    list(GET profile_fields 1 VARIANT_PROFILE_OUTPUT_TYPE)
    list(GET profile_fields 2 VARIANT_PROFILE_DATA_TYPE)
    list(GET profile_fields 3 VARIANT_DISTANCE_TYPE)
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/kernel_${VARIANT_PROFILE}_v${VARIANT_INDEX}.cu")
    if(VARIANT_UR EQUAL 0)
      set(_template "${CMAKE_CURRENT_SOURCE_DIR}/kernel_variant_shfl.cu.in")
    else()
      set(_template "${CMAKE_CURRENT_SOURCE_DIR}/kernel_variant.cu.in")
    endif()
    configure_file("${_template}" "${_out}" @ONLY)
    list(APPEND SCAMP_GENERATED_VARIANT_SOURCES "${_out}")
  endforeach()

  math(EXPR _variant_idx "${_variant_idx} + 1")
endforeach()

set(SCAMP_KVARIANTS_INIT "${_kvariants_init}")
set(SCAMP_VARIANT_DECLS "${_variant_decls}")
set(SCAMP_VARIANT_DISPATCH_BODY "${_variant_dispatch}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/kernel_variants_table.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/kernel_variants_table.h"
  @ONLY)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels_variants.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/kernels_variants.h"
  @ONLY)

add_library(gpu_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_1nn.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_1nn_index.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sum_thresh.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_matrix_summary.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_approx_all_neighbors.cu
    ${SCAMP_GENERATED_VARIANT_SOURCES}
)
target_include_directories(gpu_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${CMAKE_CURRENT_BINARY_DIR})
# Eigen3::Eigen is set up by src/core/cpu_kernel/CMakeLists.txt (which
# the parent src/core/CMakeLists.txt processes before this directory)
# as either:
#   - a header-only INTERFACE target pointing at third_party/eigen
#     (default), or
#   - the real imported target from `find_package(Eigen3 5.0.0 ...)` when
#     SCAMP_USE_EXTERNAL_EIGEN=1 (distro recipes; see environment.rst).
# Upstream Eigen still publishes itself as the `Eigen3` CMake package
# and `Eigen3::Eigen` target even at version 5.x, so either path
# produces the same link spelling here. We use Eigen's fixed-size
# arrays in the device-side compute templates for cleaner expression
# syntax than hand-unrolled C arrays; the headers are device-callable
# when --expt-relaxed-constexpr is on (set at top level).
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
