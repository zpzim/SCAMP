// Declarations for the per-(profile, variant) launch helpers. Each one is
// defined in a separately-compiled .cu file generated from
// kernel_variant.cu.in (see the foreach in
// src/core/gpu_kernel/CMakeLists.txt).
//
// Per-profile dispatcher TUs (kernel_<profile>.cu) call these via a
// SCAMP_DISPATCH_VARIANT-style cfg-switch to pick the right pre-instantiated
// variant for the autotuner's chosen KernelConfig.
//
// Adding a variant: append the tuple to kKernelVariants (kernel_config.cpp),
// add the matching SCAMP_VARIANT_DISPATCH entry in each kernel_<profile>.cu,
// and add the index to SCAMP_VARIANT_INDICES in
// src/core/gpu_kernel/CMakeLists.txt.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "common/common.h"
#include "core/kernel_common.h"

namespace SCAMP {

// Forward-declare the 6 launch helpers per profile. Each takes the same
// argument set as the parent LaunchKernel_<X> minus the KernelConfig (the
// variant is encoded in the function's name) and minus num_blocks padding.
//
// SCAMPmatch comes from common.h via core/kernel_common.h above.
#define SCAMP_DECL_VARIANTS_FOR_PROFILE(PROFILE, OUTPUT_TYPE)                  \
  SCAMPError_t LaunchVariant_##PROFILE##_v0(                                   \
      SCAMPKernelInputArgs<double> args, OUTPUT_TYPE *profile_A,               \
      OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type, bool computing_rows, \
      bool computing_cols, uint64_t blocksz, uint64_t num_blocks,              \
      uint64_t smem, cudaStream_t s);                                          \
  SCAMPError_t LaunchVariant_##PROFILE##_v1(                                   \
      SCAMPKernelInputArgs<double> args, OUTPUT_TYPE *profile_A,               \
      OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type, bool computing_rows, \
      bool computing_cols, uint64_t blocksz, uint64_t num_blocks,              \
      uint64_t smem, cudaStream_t s);                                          \
  SCAMPError_t LaunchVariant_##PROFILE##_v2(                                   \
      SCAMPKernelInputArgs<double> args, OUTPUT_TYPE *profile_A,               \
      OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type, bool computing_rows, \
      bool computing_cols, uint64_t blocksz, uint64_t num_blocks,              \
      uint64_t smem, cudaStream_t s);                                          \
  SCAMPError_t LaunchVariant_##PROFILE##_v3(                                   \
      SCAMPKernelInputArgs<double> args, OUTPUT_TYPE *profile_A,               \
      OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type, bool computing_rows, \
      bool computing_cols, uint64_t blocksz, uint64_t num_blocks,              \
      uint64_t smem, cudaStream_t s);                                          \
  SCAMPError_t LaunchVariant_##PROFILE##_v4(                                   \
      SCAMPKernelInputArgs<double> args, OUTPUT_TYPE *profile_A,               \
      OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type, bool computing_rows, \
      bool computing_cols, uint64_t blocksz, uint64_t num_blocks,              \
      uint64_t smem, cudaStream_t s);                                          \
  SCAMPError_t LaunchVariant_##PROFILE##_v5(                                   \
      SCAMPKernelInputArgs<double> args, OUTPUT_TYPE *profile_A,               \
      OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type, bool computing_rows, \
      bool computing_cols, uint64_t blocksz, uint64_t num_blocks,              \
      uint64_t smem, cudaStream_t s);                                          \
  SCAMPError_t LaunchVariant_##PROFILE##_v6(                                   \
      SCAMPKernelInputArgs<double> args, OUTPUT_TYPE *profile_A,               \
      OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type, bool computing_rows, \
      bool computing_cols, uint64_t blocksz, uint64_t num_blocks,              \
      uint64_t smem, cudaStream_t s);

SCAMP_DECL_VARIANTS_FOR_PROFILE(1NN, float)
SCAMP_DECL_VARIANTS_FOR_PROFILE(1NN_INDEX, uint64_t)
SCAMP_DECL_VARIANTS_FOR_PROFILE(SUM_THRESH, double)
SCAMP_DECL_VARIANTS_FOR_PROFILE(MATRIX_SUMMARY, float)
SCAMP_DECL_VARIANTS_FOR_PROFILE(APPROX_ALL_NEIGHBORS, SCAMPmatch)

#undef SCAMP_DECL_VARIANTS_FOR_PROFILE

// SCAMP_VARIANT_DISPATCH expands inside a LaunchKernel_<PROFILE> body to the
// cfg-switch that picks the right LaunchVariant_<PROFILE>_v<N>. The variant
// tuples MUST stay in sync with kKernelVariants in kernel_config.cpp and
// with the SCAMP_VARIANT_INDICES list in
// src/core/gpu_kernel/CMakeLists.txt.
//
// On no match, falls through to `return SCAMP_CUDA_ERROR`: this is the
// upstream's signal that IsSupportedKernelConfig let through an unsupported
// cfg, which would only happen if kKernelVariants and the dispatch table
// drifted apart.
#define SCAMP_VARIANT_DISPATCH(PROFILE)                                        \
  do {                                                                         \
    /* TEMPORARY: v0..v5 dispatch commented out during shfl draft. Re-enable   \
     * along with re-enabling SCAMP_VARIANT_TUPLES in CMakeLists.txt before    \
     * merging. */                                                             \
    /* v6: design-A "shfl" variant, ur==0 sentinel. */                         \
    if (cfg.blocks_per_sm == 2 && cfg.diags_per_thread == 2 &&                 \
        cfg.unrolled_rows == 0 && cfg.outer_unrolled_rows == 8 &&              \
        cfg.kernel_tile_iters == 8) {                                          \
      return LaunchVariant_##PROFILE##_v6(args, profile_A, profile_B, fp_type, \
                                          computing_rows, computing_cols,      \
                                          cfg.blocksz, num_blocks, smem, s);   \
    }                                                                          \
    return SCAMP_CUDA_ERROR;                                                   \
  } while (0)

}  // namespace SCAMP
