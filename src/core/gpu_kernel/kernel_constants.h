// Plain compile-time launch constants for the SCAMP GPU kernel. Kept
// CUDA-free so host translation units (e.g. kernel_config.cpp) can include
// these without pulling in nvcc-only intrinsics from kernel_gpu_utils.h.
#pragma once

namespace SCAMP {

// Number of diagonals processed per thread. The kernel covers a metadiagonal
// of width BLOCKSZ * DIAGS_PER_THREAD per thread block.
constexpr int DIAGS_PER_THREAD = 2;

// Inner-loop unroll knobs for do_iteration_fast.
//
// The fast-path kernel computes a parallelogram of distances per call:
// outer_unrolled_rows rows tall by unrolled_diags diagonals wide. Within each
// call, the row loop is sub-unrolled into batches of unrolled_rows rows; after
// each batch the per-thread column register window (inner_unrolled_cols wide)
// slides forward by unrolled_rows positions, loading unrolled_rows new column
// values from shared memory and dropping the oldest.
//
// inner_unrolled_cols is fixed by the geometry: each row batch needs
// unrolled_diags neighbouring columns per row, plus (unrolled_rows - 1) extra
// to cover the row-by-row shift.
//
// Adjust these to retune register pressure / smem bandwidth / occupancy.
constexpr int unrolled_diags = DIAGS_PER_THREAD;
constexpr int unrolled_rows = 2;
constexpr int outer_unrolled_rows = 16;
constexpr int inner_unrolled_cols = unrolled_diags + unrolled_rows - 1;
constexpr int unrolled_cols = DIAGS_PER_THREAD + outer_unrolled_rows - 1;

// Number of do_iteration_fast calls per tile.
constexpr int KERNEL_TILE_ITERS = 16;
// Height of the parallelogram a thread block processes per tile.
constexpr int KERNEL_TILE_HEIGHT = KERNEL_TILE_ITERS * outer_unrolled_rows;

// Threads per block for single-precision (SP) and double-precision (DP) paths.
constexpr int BLOCKSZ_SP = 512;
constexpr int BLOCKSZ_DP = 256;

// __launch_bounds__ second argument (occupancy hint).
constexpr int BLOCKSPERSM = 2;

constexpr int TILE_HEIGHT_SP = KERNEL_TILE_HEIGHT;
constexpr int TILE_HEIGHT_DP = KERNEL_TILE_HEIGHT;

// ---------------------------------------------------------------------------
// Autotune variants.
//
// The autotuner picks one of these (tile_height, blocks_per_sm) variants per
// (device, profile_type, precision). Each variant must be enumerated here AND
// instantiated in the LaunchDoTile dispatch in kernels_impl.h; entries that
// aren't pre-instantiated will fall back to the default at lookup time.
//
// Adding a variant: append a row to kKernelVariants in kernel_config.cpp,
// add a branch in the LaunchDoTile switch (kernels_impl.h), and the
// autotuner will start benchmarking it on the next RunAutotune() invocation.
// Removing one is the reverse, plus pruning stale cache entries.
// ---------------------------------------------------------------------------

// Alt variant: smaller tile, higher occupancy. Useful on devices where the
// default's smem footprint limits blocks_per_sm.
constexpr int KERNEL_TILE_HEIGHT_ALT = 128;
constexpr int BLOCKSPERSM_ALT = 4;

}  // namespace SCAMP
