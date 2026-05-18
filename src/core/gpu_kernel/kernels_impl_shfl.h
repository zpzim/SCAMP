// Top-level kernel and launcher for the design-A "cov-shuffle" variant.
// Mirrors kernels_impl.h's do_tile + LaunchDoTileWithGeometry but uses
// the shfl algorithm + SCAMPShflSmem layout.

#pragma once

#include <cuda.h>

#include "core/defines.h"
#include "core/kernel_common.h"
#include "kernel_gpu_utils.h"
#include "kernels_compute_shfl.h"

namespace SCAMP {

// kernels_smem.h is unwrapped (no namespace SCAMP block of its own) so it
// must be included inside the namespace. We need it for write_back /
// write_back_value -- the per-tile post-compute flush from smem to global.
#include "kernels_smem.h"  // NOLINT(build/include)

// Computes the matrix profile via the cov-shuffle algorithm. Each lane
// owns a FIXED DPT-wide column slice within the block's parallelogram
// tile. cov walks lane-to-lane via shuffle each row; lane's column block
// rotates every 32*DPT rows via update_info_shfl.
//
// Template params:
//   blocks_per_sm  — __launch_bounds__ occupancy hint
//   DiagsPerThread — diagonals per thread (column slice width)
//   OuterUnrolledRows — rows per inner-loop iteration (for compile time
//                       amortization; the shfl kernel processes 1 row at
//                       a time logically, but the loop is unrolled in
//                       chunks of OUR for code-size control)
//   KernelTileIters — outer iterations per tile (tile_height = OUR * KTI)
//   BLOCKSZ        — threads per block
//
// MUST hold: tile_height (= OUR * KTI) <= 32 * DPT for the FIRST DRAFT.
// Larger tile_height would require multiple column rotations per tile,
// which works in principle (the prototype handles it) but is left for
// a follow-up so the first version stays simple to validate.
template <typename DATA_TYPE, typename ACCUM_TYPE, typename PROFILE_OUTPUT_TYPE,
          typename PROFILE_DATA_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, SCAMPProfileType PROFILE_TYPE, int blocks_per_sm,
          int DiagsPerThread, int OuterUnrolledRows, int KernelTileIters,
          int BLOCKSZ>
__global__ void __launch_bounds__(BLOCKSZ, blocks_per_sm)
    do_tile_shfl(SCAMPKernelInputArgs<double> args,
                 PROFILE_OUTPUT_TYPE *profile_A,
                 PROFILE_OUTPUT_TYPE *profile_B) {
  constexpr int tile_height = KernelTileIters * OuterUnrolledRows;
  // tile_width is the PARALLELOGRAM width: each lane in the block rotates
  // its column slice by 32*DPT every 32*DPT rows (staggered per warp-lane),
  // so over tile_height rows the block touches columns up to
  // BLOCKSZ*DPT + tile_height - 1. Sized smem accordingly so
  // state.local_col indexing into smem.local_mp_col stays in bounds.
  constexpr int tile_width = BLOCKSZ * DiagsPerThread + tile_height;
  constexpr int warps_per_block = BLOCKSZ / 32;
  static_assert(BLOCKSZ % 32 == 0, "BLOCKSZ must be a multiple of 32");
  static_assert(tile_height <= 32 * DiagsPerThread,
                "tile_height must be <= 32 * DPT for the first draft");

  extern __shared__ char smem_raw[];
  SCAMPShflSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE, tile_width,
                tile_height, warps_per_block>
      smem(smem_raw, COMPUTE_ROWS, COMPUTE_COLS, args.opt.num_extra_operands);

  // Block geometry: matches the sliding-window kernel's 1D meta-diagonal
  // grid. Each block walks one meta-diagonal through tile-height steps.
  const unsigned int start_diag = args.exclusion_lower +
                                  (threadIdx.x * DiagsPerThread) +
                                  blockIdx.x * (blockDim.x * DiagsPerThread);
  const unsigned int meta_diagonal_idx = blockIdx.x;
  uint32_t tile_start_col =
      meta_diagonal_idx * (BLOCKSZ * DiagsPerThread) + args.exclusion_lower;
  uint32_t tile_start_row = 0;
  const unsigned int num_diags = args.n_x - args.exclusion_upper + 1;

  // Per-thread state. dfc/dgc/inormc/cov are loaded once per BLOCK
  // (lifetime extends across all tiles within the block) and rotate via
  // update_info_shfl every 32*DPT rows.
  SCAMPShflState<DATA_TYPE, DISTANCE_TYPE, DiagsPerThread> state;
  state.warpln = threadIdx.x & 31u;
  state.warpid = threadIdx.x >> 5;
  state.srcln = (state.warpln - 1u) & 31u;
  state.updates_remaining =
      state.warpln * DiagsPerThread + (DiagsPerThread - 1);
  state.global_col = tile_start_col + threadIdx.x * DiagsPerThread;
  state.local_col = threadIdx.x * DiagsPerThread;

  // Initial cov: cov(0, global_col + i) = args.cov[global_col + i].
#pragma unroll
  for (int i = 0; i < DiagsPerThread; ++i) {
    if (state.global_col + i < static_cast<uint32_t>(args.n_x)) {
      state.cov[i] = static_cast<DATA_TYPE>(args.cov[state.global_col + i]);
    } else {
      state.cov[i] = DATA_TYPE(0);
    }
  }

  // Initial column data: lane T loads its own DPT-wide slice from global.
#pragma unroll
  for (int i = 0; i < DiagsPerThread; ++i) {
    uint32_t pos = state.global_col + i;
    if (pos < static_cast<uint32_t>(args.n_x)) {
      state.dfc[i] = static_cast<DATA_TYPE>(args.dfa[pos]);
      state.dgc[i] = static_cast<DATA_TYPE>(args.dga[pos]);
      state.inormc[i] = static_cast<DATA_TYPE>(args.normsa[pos]);
    } else {
      state.dfc[i] = DATA_TYPE(0);
      state.dgc[i] = DATA_TYPE(0);
      state.inormc[i] = DATA_TYPE(0);
    }
  }
  // dfc2 / dgc2 / inormc2 are populated lazily at the start of each rotation
  // cycle inside update_info_shfl (updates_remaining == DPT - 1).
  DISTANCE_TYPE init = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  state.distc = Eigen::Array<DISTANCE_TYPE, DiagsPerThread, 1>::Constant(init);
  state.idxc = Eigen::Array<unsigned int, DiagsPerThread, 1>::Zero();

  // Tile loop. Same shape as the sliding-window do_tile but with the
  // shfl inner loop.
  while (tile_start_col < args.n_x && tile_start_row < args.n_y) {
    init_smem_shfl<decltype(smem), PROFILE_DATA_TYPE, PROFILE_OUTPUT_TYPE,
                   COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height, BLOCKSZ,
                   PROFILE_TYPE>(args, smem, profile_A, profile_B,
                                 tile_start_col, tile_start_row);

    // Reset state.local_col to its initial value. Across tiles, state.local_col
    // grew via update_info_shfl rotations; for the NEXT tile's smem column
    // profile region (which is per-tile sized, indexed 0..tile_width-1),
    // the lane is back at its leftmost slot. state.global_col is already
    // correct (rotation advanced it by 32*DPT each cycle, which matches
    // tile_start_col advancing by tile_height = 32*DPT between tiles).
    state.local_col = threadIdx.x * DiagsPerThread;

    __syncthreads();

    // FAST PATH: all of this tile's cells fit in the matrix profile range.
    // For the FIRST DRAFT we only implement the fast path; tiles needing
    // edge handling fall through to the slow path placeholder below
    // (which currently does nothing). This will report wrong matches at
    // matrix edges but is sufficient for validating the kernel's
    // correctness on the bulk of the workload.
    if (tile_start_col + tile_width < args.n_x &&
        tile_start_row + tile_height < args.n_y &&
        start_diag + DiagsPerThread <= num_diags) {
#pragma unroll
      for (int r = 0; r < tile_height; ++r) {
        do_row_shfl<PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, DISTANCE_TYPE,
                    DiagsPerThread, BLOCKSZ>(r, tile_start_row, args, state,
                                             smem, smem.cov_handoff);
      }
    } else {
      // TODO(stage 5): port do_row_edge to the shfl geometry. For the
      // draft, just skip — drives some incorrect results at tile edges.
    }

    __syncthreads();

    write_back<PROFILE_TYPE, COMPUTE_COLS, COMPUTE_ROWS, BLOCKSZ, tile_width,
               tile_height>(args, smem, tile_start_col, tile_start_row,
                            args.n_x, args.n_y, profile_A, profile_B);

    tile_start_col += tile_height;
    tile_start_row += tile_height;
    __threadfence_block();

    if (NeedsCheckIfDone(PROFILE_TYPE)) {
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

// Per-(geometry x precision) launch helper for the shfl kernel. Mirrors
// LaunchDoTileWithGeometry. The shfl variant has no UR template param, so
// the signature differs there.
template <typename PROFILE_OUTPUT_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, SCAMPProfileType PROFILE_TYPE,
          int blocks_per_sm_v, int DiagsPerThread, int OuterUnrolledRows,
          int KernelTileIters>
SCAMPError_t LaunchDoTileShflWithGeometry(
    SCAMPKernelInputArgs<double> args, PROFILE_OUTPUT_TYPE *profile_A,
    PROFILE_OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type,
    bool computing_rows, bool computing_cols, uint64_t blocksz,
    uint64_t num_blocks, uint64_t smem, cudaStream_t s) {
  dim3 block(blocksz, 1, 1);
  dim3 grid(num_blocks, 1, 1);

#define LAUNCH_PRECISION_SHFL(DATA_T, ACCUM_T, BLOCKSZ_V, COMP_ROWS,          \
                              COMP_COLS)                                      \
  do {                                                                        \
    auto kfn =                                                                \
        do_tile_shfl<DATA_T, ACCUM_T, PROFILE_OUTPUT_TYPE, PROFILE_DATA_TYPE, \
                     DISTANCE_TYPE, COMP_ROWS, COMP_COLS, PROFILE_TYPE,       \
                     blocks_per_sm_v, DiagsPerThread, OuterUnrolledRows,      \
                     KernelTileIters, BLOCKSZ_V>;                             \
    if (smem > 48u * 1024u) {                                                 \
      cudaFuncSetAttribute(reinterpret_cast<const void *>(kfn),               \
                           cudaFuncAttributeMaxDynamicSharedMemorySize,       \
                           static_cast<int>(smem));                           \
    }                                                                         \
    kfn<<<grid, block, smem, s>>>(args, profile_A, profile_B);                \
  } while (0)

#define LAUNCH_FOR_ROWCOL_MODE_SHFL(COMP_ROWS, COMP_COLS)                      \
  switch (fp_type) {                                                           \
    case PRECISION_ULTRA:                                                      \
    case PRECISION_DOUBLE:                                                     \
      LAUNCH_PRECISION_SHFL(double, double, BLOCKSZ_DP, COMP_ROWS, COMP_COLS); \
      break;                                                                   \
    case PRECISION_SINGLE:                                                     \
      LAUNCH_PRECISION_SHFL(float, float, BLOCKSZ_SP, COMP_ROWS, COMP_COLS);   \
      break;                                                                   \
    default:                                                                   \
      return SCAMP_CUDA_ERROR;                                                 \
  }

  if (computing_rows && computing_cols) {
    LAUNCH_FOR_ROWCOL_MODE_SHFL(true, true);
  } else if (computing_cols) {
    LAUNCH_FOR_ROWCOL_MODE_SHFL(false, true);
  } else if (computing_rows) {
    LAUNCH_FOR_ROWCOL_MODE_SHFL(true, false);
  }
#undef LAUNCH_FOR_ROWCOL_MODE_SHFL
#undef LAUNCH_PRECISION_SHFL
  gpuErrchk(cudaPeekAtLastError());
  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
