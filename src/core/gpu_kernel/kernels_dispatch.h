// Per-profile-type launch entry points. Each LaunchKernel_<PROFILE> function
// is defined in its own .cu file (kernel_<profile>.cu) so that the heavy
// template instantiations for each profile type can compile in parallel.
// The top-level dispatcher in kernels.cu picks one of these based on
// info()->profile_type at runtime.
//
// The cfg parameter is the autotuner-selected (or default) launch geometry;
// each entry point switches on cfg.tile_height and cfg.blocks_per_sm to pick
// the matching pre-instantiated do_tile variant.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "common/common.h"
#include "core/kernel_common.h"
#include "kernel_config.h"

namespace SCAMP {

SCAMPError_t LaunchKernel_1NN_INDEX(SCAMPKernelInputArgs<double> args,
                                    uint64_t *profile_A, uint64_t *profile_B,
                                    SCAMPPrecisionType fp_type,
                                    bool computing_rows, bool computing_cols,
                                    KernelConfig cfg, uint64_t num_blocks,
                                    uint64_t smem, cudaStream_t s);

SCAMPError_t LaunchKernel_1NN(SCAMPKernelInputArgs<double> args,
                              float *profile_A, float *profile_B,
                              SCAMPPrecisionType fp_type, bool computing_rows,
                              bool computing_cols, KernelConfig cfg,
                              uint64_t num_blocks, uint64_t smem,
                              cudaStream_t s);

SCAMPError_t LaunchKernel_SUM_THRESH(SCAMPKernelInputArgs<double> args,
                                     double *profile_A, double *profile_B,
                                     SCAMPPrecisionType fp_type,
                                     bool computing_rows, bool computing_cols,
                                     KernelConfig cfg, uint64_t num_blocks,
                                     uint64_t smem, cudaStream_t s);

SCAMPError_t LaunchKernel_MATRIX_SUMMARY(
    SCAMPKernelInputArgs<double> args, float *profile_A, float *profile_B,
    SCAMPPrecisionType fp_type, bool computing_rows, bool computing_cols,
    KernelConfig cfg, uint64_t num_blocks, uint64_t smem, cudaStream_t s);

SCAMPError_t LaunchKernel_APPROX_ALL_NEIGHBORS(
    SCAMPKernelInputArgs<double> args, SCAMPmatch *profile_A,
    SCAMPmatch *profile_B, SCAMPPrecisionType fp_type, bool computing_rows,
    bool computing_cols, KernelConfig cfg, uint64_t num_blocks, uint64_t smem,
    cudaStream_t s);

}  // namespace SCAMP
