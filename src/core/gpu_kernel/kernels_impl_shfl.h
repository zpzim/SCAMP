// Top-level kernel and launcher for the cov-shuffle variant. Mirrors
// kernels_impl.h's do_tile + LaunchDoTileWithGeometry but uses the
// shfl-based cov propagation + SCAMPShflSmem layout.

#pragma once

#include "common/cuda_to_hip.h"

#if !defined(USE_HIP)
#include <cuda.h>
#endif

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
// tile. cov walks lane-to-lane via shuffle each row; a warp of kWarpSize
// lanes traverses its kWarpSize*DPT-column span over kWarpSize rows; lane's
// column block rotates via update_info_shfl.
//
// Template params:
//   target_threads_per_sm — variant author's intended thread density. The
//                       second arg of __launch_bounds__ is derived from
//                       this via safe_bps() so the per-thread register
//                       budget stays uniform across the autotuner's
//                       blocksz sweep.
//   DiagsPerThread — diagonals per thread (column slice width)
//   OuterUnrolledRows — rows per inner-loop iteration (for compile time
//                       amortization; the shfl kernel processes 1 row at
//                       a time logically, but the loop is unrolled in
//                       chunks of OUR for code-size control)
//   KernelTileIters — outer iterations per tile (tile_height = OUR * KTI)
//   BLOCKSZ        — threads per block
//
// MUST hold: tile_height (= OUR * KTI) <= kWarpSize * DPT. A warp of
// kWarpSize lanes owns kWarpSize*DPT columns; the covariance diagonal
// walks the whole warp in kWarpSize rows. Larger tile_height would require
// multiple column rotations per tile (update_info_shfl would need to fire
// more than once per kWarpSize rows). Today the kernel only fires it once
// -- the variants in kVariants all satisfy tile_height <= 32*DPT (the
// wave32 bound, which is also <= the wave64 bound of 64*DPT), and the
// autotuner doesn't sweep configs that would violate it.
template <typename DATA_TYPE, typename ACCUM_TYPE, typename PROFILE_OUTPUT_TYPE,
          typename PROFILE_DATA_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, SCAMPProfileType PROFILE_TYPE,
          int target_threads_per_sm, int DiagsPerThread, int OuterUnrolledRows,
          int KernelTileIters, int BLOCKSZ>
__global__ void __launch_bounds__(BLOCKSZ,
                                  safe_bps(target_threads_per_sm, BLOCKSZ))
    do_tile_shfl(SCAMPKernelInputArgs<double> args,
                 PROFILE_OUTPUT_TYPE *profile_A,
                 PROFILE_OUTPUT_TYPE *profile_B) {
  constexpr int tile_height = KernelTileIters * OuterUnrolledRows;
  // tile_width is the PARALLELOGRAM width: each lane in the block rotates
  // its column slice by BLOCKSZ*DPT every BLOCKSZ*DPT rows (block-level
  // stride, staggered per block thread), so over tile_height rows the block
  // touches columns up to BLOCKSZ*DPT + tile_height - 1. Sized smem
  // accordingly so state.local_col indexing into smem.local_mp_col stays in
  // bounds.
  constexpr int tile_width = BLOCKSZ * DiagsPerThread + tile_height;
  constexpr int warps_per_block = BLOCKSZ / kWarpSize;
  static_assert(BLOCKSZ % kWarpSize == 0,
                "BLOCKSZ must be a multiple of kWarpSize (warp width)");
  static_assert(tile_height <= kWarpSize * DiagsPerThread,
                "tile_height must be <= kWarpSize * DPT: do_tile_shfl assumes "
                "update_info_shfl fires at most once per warp per tile, "
                "which holds only when tile_height does not exceed the "
                "warp's column-block span of kWarpSize * DPT rows.");

  extern __shared__ char smem_raw[];
  SCAMPShflSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE, tile_width,
                tile_height, warps_per_block>
      smem(smem_raw, COMPUTE_ROWS, COMPUTE_COLS, args.opt.num_extra_operands);

  // Per-row cross-warp cov hand-off barrier. On sm_70+ this is a real
  // cuda::barrier<thread_scope_block> in static smem (single 8-byte
  // mbarrier word, separate from the dynamic SCAMPShflSmem layout so
  // existing smem-budget computations in get_smem_shfl don't need to
  // track it). Expected arrival count = BLOCKSZ (one arrive per thread).
  //
  // do_row_shfl issues an ARRIVE right after the lane-(kWarpSize-1) publish (mid-row)
  // and a WAIT at end of row; on Ampere (sm_80+) this lowers to mbarrier
  // PTX, which is genuinely non-blocking on arrive. The publish-to-arrive
  // distance is tight (one smem store), but the arrive-to-wait window
  // covers the dist masking, merge_to_col, merge_to_row, intra-warp shfl
  // cov rotation, register shift, and (every DPT rows) update_info_shfl.
  // Warps that finish that second-half work faster don't block warps
  // still draining it -- the wait blocks only on the slowest arrival.
  //
  // On sm_60 cuda::barrier is unavailable (libcudacxx errors). ShflRowBarrier
  // degrades to an empty struct and do_row_shfl falls back to
  // __syncthreads() at end of row.
  __shared__ ShflRowBarrier row_bar;
#ifdef SCAMP_SHFL_HAS_CUDA_BARRIER
  if (threadIdx.x == 0) {
    init(&row_bar, BLOCKSZ);
  }
  __syncthreads();
#endif

  // Block geometry: matches the sliding-window kernel's 1D meta-diagonal
  // grid. Each block walks one meta-diagonal through tile-height steps.
  const unsigned int meta_diagonal_idx = blockIdx.x;
  uint32_t tile_start_col =
      meta_diagonal_idx * (BLOCKSZ * DiagsPerThread) + args.exclusion_lower;
  uint32_t tile_start_row = 0;
  const unsigned int num_diags = args.n_x - args.exclusion_upper + 1;

  // Per-thread state. dfc/dgc/inormc/cov are loaded once per BLOCK
  // (lifetime extends across all tiles within the block) and rotate via
  // update_info_shfl (block-level stride BLOCKSZ*DPT, staggered per thread).
  SCAMPShflState<DATA_TYPE, DISTANCE_TYPE, DiagsPerThread> state;
  state.warpln = threadIdx.x % static_cast<uint32_t>(kWarpSize);
  state.warpid = threadIdx.x / static_cast<uint32_t>(kWarpSize);
  state.srcln = (state.warpln + static_cast<uint32_t>(kWarpSize) - 1u) %
                static_cast<uint32_t>(kWarpSize);
  state.updates_remaining = threadIdx.x * DiagsPerThread + (DiagsPerThread - 1);

#pragma unroll
  for (int i = 0; i < DiagsPerThread; ++i) {
    state.global_col[i] = tile_start_col + threadIdx.x * DiagsPerThread + i;
    state.local_col[i] = threadIdx.x * DiagsPerThread + i;
  }

  // Initial cov: cov(0, global_col + i) = args.cov[global_col + i].
#pragma unroll
  for (int i = 0; i < DiagsPerThread; ++i) {
    if (state.global_col[i] < static_cast<uint32_t>(args.n_x)) {
      state.cov[i] = static_cast<DATA_TYPE>(args.cov[state.global_col[i]]);
    } else {
      state.cov[i] = DATA_TYPE(0);
    }
  }

  // Initial column data: lane T loads its own DPT-wide slice from global.
#pragma unroll
  for (int i = 0; i < DiagsPerThread; ++i) {
    uint32_t pos = state.global_col[i];
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
  // Column values (dfc/dgc/inormc) are populated directly from global memory
  // inside update_info_shfl when a column slot rotates.
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

    // Reset state.local_col for this tile. Across tiles, state.global_col
    // tracks the absolute column each slot owns (advanced by update_info_shfl
    // rotations during the tile); the per-tile smem column profile region is
    // indexed 0..tile_width-1 relative to tile_start_col. Recomputing
    // local_col = global_col - tile_start_col here is self-correcting and
    // independent of warp width and rotation history.
    state.local_col = state.global_col - tile_start_col;

    __syncthreads();

    // Reset per-thread distc/idxc for this tile. Cross-tile continuity
    // does NOT require pre-loading from smem.local_mp_col here: the
    // merge happens at the END of each tile inside flush_all_cols_to_smem
    // via atomicMax (MPatomicMax for the index-bearing profile types,
    // atomicAdd for SUM_THRESH), so smem.local_mp_col[col_idx] correctly
    // becomes max(prior_global, this_tile_max) without ever needing to
    // seed distc with the prior value. Starting distc at init_dist (or
    // 0 for SUM_THRESH) is functionally equivalent.
    //
    // FUTURE OPT (max-style profiles only: 1NN, 1NN_INDEX,
    // APPROX_ALL_NEIGHBORS, MATRIX_SUMMARY): pre-load
    // smem.local_mp_col[col_idx] into a per-slot "check" register and
    // pass it to fAtomicMax_check / MPatomicMax_check at flush time --
    // skipping the atomic CAS entirely when this tile's distc didn't
    // beat the prior global. Correctness requires also refreshing the
    // check inside update_info_shfl when a slot rotates mid-tile, and
    // the extra DPT registers may regress occupancy on the higher-DPT
    // shfl variants (DPT=8 SP is already register-pressured). Wins
    // would be largest on workloads where the matrix profile has
    // converged and most atomics would no-op.
    state.distc.setConstant(init_dist<DISTANCE_TYPE, PROFILE_TYPE>());
    state.idxc.setZero();

    // FAST PATH: all of this tile's cells fit in the matrix profile range.
    // SLOW PATH: fallback to handle matrix bounds and exclusion zones.
    // Determine uniform block-level condition to select fast/slow path.
    const bool fast_path =
        (tile_start_col + tile_width < args.n_x) &&
        (tile_start_row + tile_height < args.n_y) &&
        (tile_start_col + BLOCKSZ * DiagsPerThread <= num_diags) &&
        (tile_start_col >= args.exclusion_upper + tile_height);

    // NB: the row loop below is deliberately NOT #pragma unroll'd.
    // tile_height is up to 32*DPT = 256 for the DPT=8 variants, and
    // do_row_shfl ends with __syncthreads() -- unrolling would inline
    // the per-row body (cross-warp publish + read, distc merge,
    // intra-warp shfl, update_info_shfl dispatch with 8 inlined slot
    // bodies for DPT=8) 256 times per kernel-template combo, ballooning
    // compile time without buying inter-iteration ILP since the barrier
    // serializes adjacent rows anyway.
    if (fast_path) {
      for (int r = 0; r < tile_height; ++r) {
        do_row_shfl<PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, DISTANCE_TYPE,
                    DiagsPerThread, BLOCKSZ, true>(
            r, tile_start_row, args, state, smem, smem.cov_handoff, row_bar);
      }
    } else {
      // Must be executed by ALL threads in the block to prevent deadlock at
      // the per-row cuda::barrier wait.
      for (int r = 0; r < tile_height; ++r) {
        do_row_shfl<PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, DISTANCE_TYPE,
                    DiagsPerThread, BLOCKSZ, false>(
            r, tile_start_row, args, state, smem, smem.cov_handoff, row_bar);
      }
    }

    if constexpr (COMPUTE_COLS) {
      flush_all_cols_to_smem<PROFILE_TYPE>(state, smem);
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
//
// default_blocksz_sp comes from the variant tuple and pins the variant
// author's intended SP thread density: target threads/SM (SP) =
// blocks_per_sm_v * default_blocksz_sp. The DP target is half of that:
// DP uses 2x the registers per thread on Ampere/Ada, so halving the
// target threads/SM keeps the per-thread register budget stable across
// precisions. The /2 is hardcoded here (not a per-variant knob) so the
// ratio cannot drift between variants.
//
// The kernel's __launch_bounds__ then derives a per-(BLOCKSZ-instantiation)
// bps from the precision-specific target via safe_bps(), so register
// pressure stays constant across the autotuner's blocksz sweep instead of
// imploding when blocksz > the variant's default.
template <typename PROFILE_OUTPUT_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, SCAMPProfileType PROFILE_TYPE,
          int blocks_per_sm_v, int DiagsPerThread, int OuterUnrolledRows,
          int KernelTileIters, int default_blocksz_sp>
SCAMPError_t LaunchDoTileShflWithGeometry(
    SCAMPKernelInputArgs<double> args, PROFILE_OUTPUT_TYPE *profile_A,
    PROFILE_OUTPUT_TYPE *profile_B, SCAMPPrecisionType fp_type,
    bool computing_rows, bool computing_cols, uint64_t blocksz,
    uint64_t num_blocks, uint64_t smem, cudaStream_t s) {
  static_assert(default_blocksz_sp >= 2,
                "default_blocksz_sp must be >= 2 so the implicit DP value "
                "(default_blocksz_sp / 2) is >= 1.");
  constexpr int default_blocksz_dp = default_blocksz_sp / 2;
  dim3 block(blocksz, 1, 1);
  dim3 grid(num_blocks, 1, 1);

#define LAUNCH_PRECISION_SHFL_AT_BLOCKSZ(DATA_T, ACCUM_T, TARGET_THREADS,     \
                                         BLOCKSZ_V, COMP_ROWS, COMP_COLS)     \
  do {                                                                        \
    auto kfn =                                                                \
        do_tile_shfl<DATA_T, ACCUM_T, PROFILE_OUTPUT_TYPE, PROFILE_DATA_TYPE, \
                     DISTANCE_TYPE, COMP_ROWS, COMP_COLS, PROFILE_TYPE,       \
                     TARGET_THREADS, DiagsPerThread, OuterUnrolledRows,       \
                     KernelTileIters, BLOCKSZ_V>;                             \
    if (smem > 48u * 1024u) {                                                 \
      cudaFuncSetAttribute(reinterpret_cast<const void *>(kfn),               \
                           cudaFuncAttributeMaxDynamicSharedMemorySize,       \
                           static_cast<int>(smem));                           \
    }                                                                         \
    kfn<<<grid, block, smem, s>>>(args, profile_A, profile_B);                \
  } while (0)

#define LAUNCH_PRECISION_SHFL(DATA_T, ACCUM_T, DEFAULT_BSZ, COMP_ROWS,       \
                              COMP_COLS)                                     \
  do {                                                                       \
    constexpr int target_threads = blocks_per_sm_v * (DEFAULT_BSZ);          \
    if (blocksz == 64) {                                                     \
      LAUNCH_PRECISION_SHFL_AT_BLOCKSZ(DATA_T, ACCUM_T, target_threads, 64,  \
                                       COMP_ROWS, COMP_COLS);                \
    } else if (blocksz == 128) {                                             \
      LAUNCH_PRECISION_SHFL_AT_BLOCKSZ(DATA_T, ACCUM_T, target_threads, 128, \
                                       COMP_ROWS, COMP_COLS);                \
    } else if (blocksz == 256) {                                             \
      LAUNCH_PRECISION_SHFL_AT_BLOCKSZ(DATA_T, ACCUM_T, target_threads, 256, \
                                       COMP_ROWS, COMP_COLS);                \
    } else {                                                                 \
      LAUNCH_PRECISION_SHFL_AT_BLOCKSZ(DATA_T, ACCUM_T, target_threads, 512, \
                                       COMP_ROWS, COMP_COLS);                \
    }                                                                        \
  } while (0)

#define LAUNCH_FOR_ROWCOL_MODE_SHFL(COMP_ROWS, COMP_COLS)                  \
  switch (fp_type) {                                                       \
    case PRECISION_ULTRA:                                                  \
    case PRECISION_DOUBLE:                                                 \
      LAUNCH_PRECISION_SHFL(double, double, default_blocksz_dp, COMP_ROWS, \
                            COMP_COLS);                                    \
      break;                                                               \
    case PRECISION_SINGLE:                                                 \
      LAUNCH_PRECISION_SHFL(float, float, default_blocksz_sp, COMP_ROWS,   \
                            COMP_COLS);                                    \
      break;                                                               \
    default:                                                               \
      return SCAMP_CUDA_ERROR;                                             \
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
#undef LAUNCH_PRECISION_SHFL_AT_BLOCKSZ
  gpuErrchk(cudaPeekAtLastError());
  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
