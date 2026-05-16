// Plain compile-time launch constants for the SCAMP GPU kernel. Kept
// CUDA-free so host translation units (e.g. kernel_config.cpp) can include
// these without pulling in nvcc-only intrinsics from kernel_gpu_utils.h.
#pragma once

namespace SCAMP {

// Tile height used by the do_tile<...> kernel.
constexpr int KERNEL_TILE_HEIGHT = 200;

// Number of diagonals processed per thread. Wired into the kernel via the
// cov1..cov4 register names in SCAMPThreadInfo; changing this value requires
// a kernel rewrite, not just retuning.
constexpr int DIAGS_PER_THREAD = 4;

// Threads per block for single-precision (SP) and double-precision (DP) paths.
constexpr int BLOCKSZ_SP = 512;
constexpr int BLOCKSZ_DP = 256;

// __launch_bounds__ second argument (occupancy hint).
constexpr int BLOCKSPERSM = 2;

constexpr int TILE_HEIGHT_SP = KERNEL_TILE_HEIGHT;
constexpr int TILE_HEIGHT_DP = KERNEL_TILE_HEIGHT;

}  // namespace SCAMP
