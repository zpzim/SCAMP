// Templated implementation of the SCAMP GPU "do_tile" kernel and the
// LaunchDoTile<...> dispatcher helper. This header is included ONLY by the
// per-profile-type translation units (kernel_1nn_index.cu, kernel_1nn.cu,
// kernel_sum_thresh.cu, kernel_matrix_summary.cu,
// kernel_approx_all_neighbors.cu) so each .cu file instantiates the template
// for exactly one profile type rather than all five being compiled serially in
// a single TU.
#pragma once

#include <cuda.h>

#include "core/defines.h"
#include "core/kernel_common.h"
#include "kernel_gpu_utils.h"

namespace SCAMP {

// kernels_compute.h and kernels_smem.h are unwrapped (no namespace SCAMP
// block of their own), so they must be included inside the namespace.
#include "kernels_compute.h"  // NOLINT(build/include)
#include "kernels_smem.h"     // NOLINT(build/include)

// Computes the matrix profile given the sliding dot products for the first
// query and the precomputed data statistics.
//
// Geometry template params (all variant-axes):
//   blocks_per_sm       __launch_bounds__ occupancy hint
//   DiagsPerThread      diagonals processed per thread (cov array width)
//   UnrolledRows        inner-loop row batch
//   OuterUnrolledRows   rows per do_iteration_fast call
//   KernelTileIters     do_iteration_fast calls per tile
//   tile_height = KernelTileIters * OuterUnrolledRows (derived)
//   BLOCKSZ             threads per block (precision-tied today)
template <typename DATA_TYPE, typename ACCUM_TYPE, typename PROFILE_OUTPUT_TYPE,
          typename PROFILE_DATA_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, SCAMPProfileType PROFILE_TYPE, int blocks_per_sm,
          int DiagsPerThread, int UnrolledRows, int OuterUnrolledRows,
          int KernelTileIters, int BLOCKSZ>
__global__ void __launch_bounds__(BLOCKSZ, blocks_per_sm)
    do_tile(SCAMPKernelInputArgs<double> args, PROFILE_OUTPUT_TYPE *profile_A,
            PROFILE_OUTPUT_TYPE *profile_B) {
  constexpr int tile_height = KernelTileIters * OuterUnrolledRows;
  constexpr int tile_width = tile_height + BLOCKSZ * DiagsPerThread;

  SCAMPThreadInfo<ACCUM_TYPE, DiagsPerThread> thread_info;

  extern __shared__ char smem_raw[];

  // Wrap the shared memory in a struct that exposes each region as an
  // Eigen::Map<Eigen::Array<..., N, 1>>, so callees can use .segment<>()
  // expressions instead of hand-unrolled raw-pointer loads.
  SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE, tile_width, tile_height>
      smem(smem_raw, COMPUTE_ROWS, COMPUTE_COLS, args.opt.num_extra_operands);

  // Find the starting diagonal of the distance matrix
  const unsigned int start_diag = args.exclusion_lower +
                                  (threadIdx.x * DiagsPerThread) +
                                  blockIdx.x * (blockDim.x * DiagsPerThread);

  // This is the index of the meta-diagonal that this thread block will work on
  const unsigned int meta_diagonal_idx = blockIdx.x;

  // The first diagonals constitute a trivial match between the same
  // subsequence, we must exclude these from the calculation according to
  // args.exclusion_lower
  uint32_t tile_start_col =
      meta_diagonal_idx * (BLOCKSZ * DiagsPerThread) + args.exclusion_lower;
  uint32_t tile_start_row = 0;

  // Initialize the column and row position of the current thread
  thread_info.global_col = tile_start_col + threadIdx.x * DiagsPerThread;
  thread_info.global_row = 0;

  // num_diags is the number of diagonals in the distance matrix, less any
  // diagonals at the end we are not computing. The +1 is required for the
  // small-window-size correctness fix (see commit 3aef7d0 / issue #135).
  const unsigned int num_diags = args.n_x - args.exclusion_upper + 1;

  // Load the first dot product values
  for (int i = 0; i < DiagsPerThread && thread_info.global_col + i < args.n_x;
       ++i) {
    thread_info.cov[i] = args.cov[thread_info.global_col + i];
  }

  while (tile_start_col < args.n_x && tile_start_row < args.n_y) {
    // Initialize the next tile's shared memory
    init_smem<decltype(smem), PROFILE_DATA_TYPE, PROFILE_OUTPUT_TYPE,
              COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height, BLOCKSZ,
              PROFILE_TYPE>(args, smem, profile_A, profile_B, tile_start_col,
                            tile_start_row);
    thread_info.local_col = threadIdx.x * DiagsPerThread;
    thread_info.local_row = 0;

    // Start of new tile, sync so we don't have data races with shared memory
    // initialization
    __syncthreads();

    // There are 2 pathways here, most of the time we take the fast path (top),
    // the last tile in every thread-block will take the slower path (bottom)
    if (tile_start_col + tile_width < args.n_x &&
        tile_start_row + tile_height < args.n_y &&
        start_diag + DiagsPerThread <= num_diags) {
      // Fast Path
      while (thread_info.local_row < tile_height) {
        do_iteration_fast<PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                          DISTANCE_TYPE, DiagsPerThread, UnrolledRows,
                          OuterUnrolledRows>(args, thread_info, smem);
      }
    } else if (start_diag < num_diags) {
      // Slow Path: one row at a time, with bound-checked per-iter cov updates
      // handled inside do_row_edge.
      while (thread_info.global_col < args.n_x &&
             thread_info.global_row < args.n_y &&
             thread_info.local_row < tile_height) {
        do_row_edge<PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, DISTANCE_TYPE,
                    DiagsPerThread>(args, thread_info, smem, start_diag,
                                    num_diags);
        ++thread_info.global_col;
        ++thread_info.global_row;
        ++thread_info.local_col;
        ++thread_info.local_row;
      }
    }

    // After this sync, the caches will be updated with the best so far values
    // for this tile
    __syncthreads();

    // Write back our best-so-far computed for this tile to global memory
    write_back<PROFILE_TYPE, COMPUTE_COLS, COMPUTE_ROWS, BLOCKSZ, tile_width,
               tile_height>(args, smem, tile_start_col, tile_start_row,
                            args.n_x, args.n_y, profile_A, profile_B);

    // Update the tile position
    tile_start_col += tile_height;
    tile_start_row += tile_height;

    // Make sure our updates were committed before we pull in the next tile
    __threadfence_block();

    if (NeedsCheckIfDone(PROFILE_TYPE)) {
      // Copy the latest value of the profile length to shared memory
      if (threadIdx.x == 0) {
        *smem.profile_a_length = *args.profile_a_length;
        *smem.profile_b_length = *args.profile_b_length;
      }

      __syncthreads();

      if (*smem.profile_a_length > args.max_matches_per_tile ||
          *smem.profile_b_length > args.max_matches_per_tile) {
        break;
      }
    }
  }
}

// Per-(geometry x precision) launch helper. Each per-profile-type .cu file
// instantiates this template once per enumerated variant, then LaunchDoTile
// picks among the instantiations based on the autotuner's chosen
// KernelConfig.
//
// Geometry template params correspond to one row of kKernelVariants (see
// kernel_config.cpp). Two precisions only: SINGLE (DATA=ACCUM=float) and
// DOUBLE/ULTRA (DATA=ACCUM=double). MIXED was dropped because it was
// uniformly slower than DOUBLE in practice -- it kept DP's accumulator
// cost without DP's smem footprint reduction.
template <typename PROFILE_OUTPUT_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, SCAMPProfileType PROFILE_TYPE,
          int blocks_per_sm_v, int DiagsPerThread, int UnrolledRows,
          int OuterUnrolledRows, int KernelTileIters>
SCAMPError_t LaunchDoTileWithGeometry(SCAMPKernelInputArgs<double> args,
                                      PROFILE_OUTPUT_TYPE *profile_A,
                                      PROFILE_OUTPUT_TYPE *profile_B,
                                      SCAMPPrecisionType fp_type,
                                      bool computing_rows, bool computing_cols,
                                      uint64_t blocksz, uint64_t num_blocks,
                                      uint64_t smem, cudaStream_t s) {
  dim3 block(blocksz, 1, 1);
  dim3 grid(num_blocks, 1, 1);
  // Expand the 3 row/col modes x 2 precisions = 6 do_tile<...> instantiations.
  // The macro keeps the dispatch table compact; each LAUNCH_PRECISION call
  // emits one nvcc <<<>>> kernel-launch line plus a cudaFuncSetAttribute
  // call to opt into >48KB dynamic smem if the variant demands it. The
  // default per-block dynamic smem cap is 48KB on sm_8.x; high-OUR
  // variants (e.g. variant 5 at OUR=16, KTI=16, tile_height=256) need ~50KB
  // for SP self-join and would otherwise fail with cudaErrorInvalidValue.
  // The opt-in is sticky per kernel function pointer, so repeated launches
  // pay only the first call.
#define LAUNCH_PRECISION_AT_BLOCKSZ(DATA_T, ACCUM_T, BLOCKSZ_V, COMP_ROWS,  \
                                    COMP_COLS)                             \
  do {                                                                     \
    auto kfn =                                                             \
        do_tile<DATA_T, ACCUM_T, PROFILE_OUTPUT_TYPE, PROFILE_DATA_TYPE,   \
                DISTANCE_TYPE, COMP_ROWS, COMP_COLS, PROFILE_TYPE,         \
                blocks_per_sm_v, DiagsPerThread, UnrolledRows,             \
                OuterUnrolledRows, KernelTileIters, BLOCKSZ_V>;            \
    if (smem > 48u * 1024u) {                                              \
      cudaFuncSetAttribute(reinterpret_cast<const void *>(kfn),            \
                           cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                           static_cast<int>(smem));                        \
    }                                                                      \
    kfn<<<grid, block, smem, s>>>(args, profile_A, profile_B);             \
  } while (0)

// Dispatch the sliding-window kernel on the runtime blocksz, mirroring the
// shfl variant. Each blocksz value yields a distinct do_tile<...> template
// instantiation; the 4-way switch lets the autotuner sweep blocksz without
// recompiling. Same as v6, BLOCKSZ values are restricted to the
// IsSupportedKernelConfig whitelist (64/128/256/512).
#define LAUNCH_PRECISION(DATA_T, ACCUM_T, COMP_ROWS, COMP_COLS)            \
  do {                                                                     \
    if (blocksz == 64) {                                                   \
      LAUNCH_PRECISION_AT_BLOCKSZ(DATA_T, ACCUM_T, 64, COMP_ROWS,          \
                                  COMP_COLS);                              \
    } else if (blocksz == 128) {                                           \
      LAUNCH_PRECISION_AT_BLOCKSZ(DATA_T, ACCUM_T, 128, COMP_ROWS,         \
                                  COMP_COLS);                              \
    } else if (blocksz == 256) {                                           \
      LAUNCH_PRECISION_AT_BLOCKSZ(DATA_T, ACCUM_T, 256, COMP_ROWS,         \
                                  COMP_COLS);                              \
    } else {                                                               \
      LAUNCH_PRECISION_AT_BLOCKSZ(DATA_T, ACCUM_T, 512, COMP_ROWS,         \
                                  COMP_COLS);                              \
    }                                                                      \
  } while (0)

#define LAUNCH_FOR_ROWCOL_MODE(COMP_ROWS, COMP_COLS)              \
  switch (fp_type) {                                              \
    case PRECISION_ULTRA:                                         \
    case PRECISION_DOUBLE:                                        \
      LAUNCH_PRECISION(double, double, COMP_ROWS, COMP_COLS);     \
      break;                                                      \
    case PRECISION_SINGLE:                                        \
      LAUNCH_PRECISION(float, float, COMP_ROWS, COMP_COLS);       \
      break;                                                      \
    default:                                                      \
      return SCAMP_CUDA_ERROR;                                    \
  }

  if (computing_rows && computing_cols) {
    LAUNCH_FOR_ROWCOL_MODE(true, true);
  } else if (computing_cols) {
    LAUNCH_FOR_ROWCOL_MODE(false, true);
  } else if (computing_rows) {
    LAUNCH_FOR_ROWCOL_MODE(true, false);
  }
#undef LAUNCH_FOR_ROWCOL_MODE
#undef LAUNCH_PRECISION
#undef LAUNCH_PRECISION_AT_BLOCKSZ
  gpuErrchk(cudaPeekAtLastError());
  return SCAMP_NO_ERROR;
}

// LaunchDoTile (the cfg switch over enumerated variants) used to live here
// as a template that each kernel_<X>.cu instantiated -- pulling all 6
// variants' LaunchDoTileWithGeometry instantiations into one TU. That made
// each kernel_<X>.cu compile 36 do_tile bodies serially.
//
// Post-split the cfg switch lives in kernels_variants.h as a
// SCAMP_VARIANT_DISPATCH macro inside each per-profile dispatcher .cu
// file, and each variant's LaunchDoTileWithGeometry instantiation lives in
// its own generated kernel_<profile>_v<N>.cu (via configure_file from
// kernel_variant.cu.in). 30 small files instead of 5 big ones.

}  // namespace SCAMP
