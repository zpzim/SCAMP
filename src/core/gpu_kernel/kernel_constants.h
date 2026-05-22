// Plain compile-time launch constants for the SCAMP GPU kernel. Kept
// CUDA-free so host translation units (e.g. kernel_config.cpp) can include
// these without pulling in nvcc-only intrinsics from kernel_gpu_utils.h.
//
// The kernel inner-loop shape knobs (diags_per_thread, unrolled_rows,
// outer_unrolled_rows, kernel_tile_iters, blocks_per_sm) vary per
// autotune variant -- their tuples live in SCAMP_VARIANT_TUPLES (see
// src/core/gpu_kernel/CMakeLists.txt), and each tuple gets baked into a
// LaunchDoTileWithGeometry<...> template instantiation generated from
// kernel_variant.cu.in.
#pragma once

namespace SCAMP {

// Threads per block, tied to precision. (Not a variant axis on its own;
// the autotuner sweeps a separate blocksz axis ({64,128,256,512}) per
// variant, but precision-defaulted variants fall back to these.)
constexpr int BLOCKSZ_SP = 512;
constexpr int BLOCKSZ_DP = 256;

}  // namespace SCAMP
