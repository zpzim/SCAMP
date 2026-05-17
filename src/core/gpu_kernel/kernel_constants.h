// Plain compile-time launch constants for the SCAMP GPU kernel. Kept
// CUDA-free so host translation units (e.g. kernel_config.cpp) can include
// these without pulling in nvcc-only intrinsics from kernel_gpu_utils.h.
//
// The kernel inner-loop shape knobs (diags_per_thread, unrolled_rows,
// outer_unrolled_rows, kernel_tile_iters, blocks_per_sm) are NOT defined
// here as global constants -- they vary per autotune variant. The variant
// table lives in kernel_config.cpp; each entry's tuple gets baked into one
// of the LaunchDoTileWithGeometry<...> template instantiations in
// kernels_impl.h. This file only holds the precision-tied constants and
// the values used by GetDefaultKernelConfig() as the variant-0 fallback.
#pragma once

namespace SCAMP {

// Threads per block, tied to precision. (Not a variant axis today; varying
// blocksz would require pre-instantiating do_tile for each value too.)
constexpr int BLOCKSZ_SP = 512;
constexpr int BLOCKSZ_DP = 256;

// Default variant initializers, also the values GetDefaultKernelConfig()
// emits as variant 0.
//
// DEFAULT_DIAGS_PER_THREAD: how many distance-matrix diagonals each thread
//   processes per fast-path iteration.
// DEFAULT_UNROLLED_ROWS: inner-loop row batch size (drives the per-thread
//   register-window slide rate).
// DEFAULT_OUTER_UNROLLED_ROWS: rows processed per do_iteration_fast call.
// DEFAULT_KERNEL_TILE_ITERS: do_iteration_fast calls per tile.
//   (tile_height = DEFAULT_KERNEL_TILE_ITERS * DEFAULT_OUTER_UNROLLED_ROWS)
// DEFAULT_BLOCKSPERSM: __launch_bounds__ occupancy hint.
constexpr int DEFAULT_DIAGS_PER_THREAD = 2;
constexpr int DEFAULT_UNROLLED_ROWS = 2;
constexpr int DEFAULT_OUTER_UNROLLED_ROWS = 16;
constexpr int DEFAULT_KERNEL_TILE_ITERS = 16;
constexpr int DEFAULT_BLOCKSPERSM = 2;

}  // namespace SCAMP
