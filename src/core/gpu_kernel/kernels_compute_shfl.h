// "cov-shuffle" GPU kernel: per-row covariance values walk lane-to-lane
// via warp shuffles rather than being held in shared memory.
//
// Algorithm (per warp, per row of work):
//   - Lane T owns a FIXED DPT-wide column slice. Ownership rolls over every
//     32*DPT rows via update_info_shfl, which reads the next 32*DPT-block
//     directly from global memory (no smem column buffer).
//   - cov[i] at row r tracks cov(r, lane_T_col + i) on diagonal
//     lane_T_col + i - r.
//   - Per row: compute dist + update cov in place (cov(r, c) -> cov(r+1, c+1)).
//   - Per row: shift cov[i] right within the lane and shuffle in
//     lane (T-1)'s post-update cov[DPT-1] into cov[0]. Net effect: each
//     diagonal's cov walks one slot per row through the warp.
//
// Cross-warp boundary handling:
//   - Lane 31 of warp k publishes its post-update cov[DPT-1] into a tiny
//     smem hand-off region.
//   - Lane 0 of warp k > 0 reads warp k-1's published value.
//   - One per-row block-scope barrier, with double-buffering of the
//     hand-off slot so publish and read in consecutive rows don't race.
//     The barrier is a cuda::barrier arrive/wait pair on sm_70+ and a
//     plain __syncthreads() on sm_60 (libcudacxx's cuda/barrier header
//     hard-errors on sm_60).
//   - Lane 0 of warp 0 has no predecessor; its cov[0] is junk after row 0
//     and is masked from distc/distr updates via the slot_valid check.
//
// Slot validity mask: cell (r, lane_T_col + i) is in the BLOCK's
// meta-diagonal range iff state.local_col + i >= r (block-local diagonal
// >= 0). Cells that fail get dist = init_dist so subsequent merges discard
// them.
//
// Smem layout: this kernel does NOT use df_col / dg_col / inorm_col (so
// SCAMPShflSmem drops those regions). Row data still uses smem (broadcast
// across all lanes of all warps is efficient). A new cov_handoff_buf
// region of 2 * warps_per_block scalars carries the cross-warp hand-off.
//
// PROFILE_TYPE specialization: distc/distr updates and per-row atomics
// follow the same per-profile dispatch as the sliding-window kernel but
// are inlined here rather than calling merge_to_column/merge_to_row (those
// expect a sliding-window-shaped distc array of width DPT + OUR - 1; the
// shfl variant's distc is just DPT wide). For SUM_THRESH the row update
// uses a warp-shuffle reduction so we issue one atomicAdd per warp per row
// instead of one per lane.

#pragma once

#include <cuda.h>

// cuda/barrier requires sm_70+ (libcudacxx hard-errors on older arches).
// We gate the cuda::barrier arrive/wait split path on __CUDA_ARCH__ >= 700;
// sm_60 (Pascal) and older fall back to a plain __syncthreads() per row.
// The host-compilation pass has __CUDA_ARCH__ undefined and must also see
// the include so launch-helper code can name the type in template params,
// so we include it whenever the arch is undefined (host) or >= 700.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#include <cuda/barrier>
#define SCAMP_SHFL_HAS_CUDA_BARRIER 1
#endif

#include "core/defines.h"
#include "core/kernel_common.h"
#include "kernel_gpu_utils.h"

namespace SCAMP {

// Wrapper type used in do_row_shfl's signature and the per-block __shared__
// declaration. On sm_70+ it is cuda::barrier<thread_scope_block> (the real
// mbarrier-backed primitive). On sm_60 the cuda/barrier header is
// unavailable, so we substitute an empty struct -- the kernel falls back
// to __syncthreads() and never reads the wrapper's value, but the type
// still needs to exist so the function signatures and the __shared__
// declaration parse.
#ifdef SCAMP_SHFL_HAS_CUDA_BARRIER
using ShflRowBarrier = cuda::barrier<cuda::thread_scope_block>;
#else
struct ShflRowBarrier {};
#endif

// Per-thread register state for the shfl kernel. The sliding-window kernel
// uses SCAMPThreadInfo; this is the shfl-specific extension.
//
//   cov                 — running accumulator. Shuffles right within warp
//                         each row.
//   dfc / dgc / inormc  — the lane's CURRENT column data. Each slot's entry
//                         is re-fetched from global memory at rotation
//                         time inside update_info_shfl (no separate
//                         staging array; the L1/TEX cache absorbs the
//                         repeat reads when adjacent lanes rotate close
//                         in time).
//   distc / idxc        — per-lane per-column best-so-far. Flushed to smem
//                         when the slot rotates.
//   updates_remaining   — rotation countdown. Initialized staggered per
//                         lane (warpln * DPT + DPT - 1) so lane 0 of each
//                         warp rotates first.
//   warpln / warpid /
//     srcln             — cached lane / warp identifiers.
//   global_col /
//     local_col         — per-slot global / block-local column anchor.
//                         Per-slot (not scalar) because the staggered
//                         rotation has each slot's column block advance
//                         independently within a cycle.
//
// Register accounting (per thread):
//   5 T-typed arrays of DPT (cov, dfc, dgc, inormc, distc)
//   3 uint32_t arrays of DPT (idxc, global_col, local_col)
//   For T = float (SP): 8 * DPT 32-bit registers from arrays.
//   For T = double (DP): 12 * DPT 32-bit registers from arrays
//     (doubles are 64-bit, taking 2 32-bit reg slots).
// Plus ~10 scalar regs. To fit bps=8 with blocksz=128 (the 65536-regs /
// 8 / 128 = 64 regs/thread ceiling), DPT*8 + 10 must stay <= 64 for SP,
// implying DPT <= 6. DPT=8 needs bps=4 for SP; for DP, DPT=4 is already
// near the bps=8 ceiling so DPT=8 DP needs bps=2.
template <typename T, typename DistType, int DPT>
struct SCAMPShflState {
  Eigen::Array<T, DPT, 1> cov;
  Eigen::Array<T, DPT, 1> dfc, dgc, inormc;
  Eigen::Array<DistType, DPT, 1> distc;
  Eigen::Array<unsigned int, DPT, 1> idxc;
  int updates_remaining;
  uint32_t warpln;
  uint32_t warpid;
  uint32_t srcln;
  Eigen::Array<uint32_t, DPT, 1> global_col;
  Eigen::Array<uint32_t, DPT, 1> local_col;
  // MATRIX_SUMMARY register accumulator (see SCAMPThreadInfo): running max for
  // the current output cell; ms_cell is the linearized global cell index, -1
  // when empty.
  int ms_cell;
  float ms_max;
};

// One rotation step. Called when updates_remaining < DPT. Flushes one
// column slot's distc/idxc atomically to smem, swaps in the next-block
// stage, and resets distc.
//
// At the START of a rotation cycle (updates_remaining == DPT - 1), bulk-
// loads the next 32*DPT-block from global into dfc2/dgc2/inormc2. Reads
// are coalesced-ish: 32 lanes × DPT consecutive doubles per lane = the
// warp covers a contiguous span with a DPT-stride pattern between lanes.
// The L1 cache absorbs the stride; total cost is ~ceil((32 * DPT *
// sizeof(T)) / 128) cache-line fetches per warp per rotation per data
// array.
template <int updates_remaining, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE, bool FAST_PATH, typename T,
          typename DistType, int DPT, typename DerivedSmem>
__device__ inline void update_info_shfl(
    const SCAMPKernelInputArgs<double> &args,
    SCAMPShflState<T, DistType, DPT> &state, DerivedSmem &smem) {
  static_assert(updates_remaining >= 0 && updates_remaining < DPT,
                "updates_remaining must index a column slot");

  constexpr int BLOCKSZ = DerivedSmem::BLOCKSZ;
  constexpr int col_to_update = (DPT - 1) - updates_remaining;

  // Flush this slot's accumulated distc/idxc to the smem column profile.
  // MATRIX_SUMMARY keeps no per-column profile (it accumulates per cell into
  // the smem grid via the register accumulator), so skip this flush -- its
  // local_mp_col region is repurposed as the cell grid.
  if constexpr (COMPUTE_COLS && PROFILE_TYPE != PROFILE_TYPE_MATRIX_SUMMARY) {
    if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
      fAtomicMax<ATOMIC_BLOCK>(
          smem.local_mp_col.data() + state.local_col[col_to_update],
          state.distc[col_to_update]);
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                         PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                         PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
      MPatomicMax<ATOMIC_BLOCK>(
          reinterpret_cast<uint64_t *>(smem.local_mp_col.data()) +
              state.local_col[col_to_update],
          state.distc[col_to_update], state.idxc[col_to_update]);
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
      do_atomicAdd<double, ATOMIC_BLOCK>(
          smem.local_mp_col.data() + state.local_col[col_to_update],
          static_cast<double>(state.distc[col_to_update]));
    }
  }

  // Load the next block's column data directly from global memory.
  uint32_t pos = state.global_col[col_to_update] + BLOCKSZ * DPT;
  if constexpr (FAST_PATH) {
    state.dfc[col_to_update] = static_cast<T>(args.dfa[pos]);
    state.dgc[col_to_update] = static_cast<T>(args.dga[pos]);
    state.inormc[col_to_update] = static_cast<T>(args.normsa[pos]);
  } else {
    if (pos < static_cast<uint32_t>(args.n_x)) {
      state.dfc[col_to_update] = static_cast<T>(args.dfa[pos]);
      state.dgc[col_to_update] = static_cast<T>(args.dga[pos]);
      state.inormc[col_to_update] = static_cast<T>(args.normsa[pos]);
    } else {
      state.dfc[col_to_update] = T(0);
      state.dgc[col_to_update] = T(0);
      state.inormc[col_to_update] = T(0);
    }
  }

  state.distc[col_to_update] = init_dist<DistType, PROFILE_TYPE>();
  state.idxc[col_to_update] = 0;

  state.global_col[col_to_update] += BLOCKSZ * DPT;
  state.local_col[col_to_update] += BLOCKSZ * DPT;

  if constexpr (updates_remaining == 0) {
    state.updates_remaining = BLOCKSZ * DPT;
  }
}

// Runtime dispatch to the compile-time-templated update_info_shfl body
// corresponding to the current updates_remaining value. The if-constexpr
// branches collapse for small DPT.
//
// MUST cover the full [0, DPT) range; the caller fires this whenever
// state.updates_remaining < DPT, and any missing value would leave the
// corresponding slot un-rotated -- which silently de-syncs state.global_col
// against tile_start_col across tile boundaries and eventually underflows
// the per-tile state.local_col reset (manifests as a 0xfffff... smem OOB
// in flush_all_cols_to_smem).
template <bool COMPUTE_COLS, SCAMPProfileType PROFILE_TYPE, bool FAST_PATH,
          typename T, typename DistType, int DPT, typename DerivedSmem>
__device__ inline void do_update_info_shfl(
    const SCAMPKernelInputArgs<double> &args,
    SCAMPShflState<T, DistType, DPT> &state, DerivedSmem &smem) {
  // The runtime dispatch ladder below has explicit cases for
  // updates_remaining = 0..7 and MUST cover the full [0, DPT) range that
  // the caller can fire with. Bump the ladder + this assert in lockstep
  // if a variant ever needs DPT > 8.
  static_assert(DPT <= 8,
                "do_update_info_shfl only dispatches updates_remaining = 0..7; "
                "extend the ladder before bumping DPT past 8 (otherwise "
                "slots 0..(DPT-9) are silently skipped, which de-syncs "
                "state.global_col against tile_start_col across tiles).");
  if constexpr (DPT >= 8) {
    if (state.updates_remaining == 7) {
      update_info_shfl<7, COMPUTE_COLS, PROFILE_TYPE, FAST_PATH>(args, state,
                                                                 smem);
      return;
    }
  }
  if constexpr (DPT >= 7) {
    if (state.updates_remaining == 6) {
      update_info_shfl<6, COMPUTE_COLS, PROFILE_TYPE, FAST_PATH>(args, state,
                                                                 smem);
      return;
    }
  }
  if constexpr (DPT >= 6) {
    if (state.updates_remaining == 5) {
      update_info_shfl<5, COMPUTE_COLS, PROFILE_TYPE, FAST_PATH>(args, state,
                                                                 smem);
      return;
    }
  }
  if constexpr (DPT >= 5) {
    if (state.updates_remaining == 4) {
      update_info_shfl<4, COMPUTE_COLS, PROFILE_TYPE, FAST_PATH>(args, state,
                                                                 smem);
      return;
    }
  }
  if constexpr (DPT >= 4) {
    if (state.updates_remaining == 3) {
      update_info_shfl<3, COMPUTE_COLS, PROFILE_TYPE, FAST_PATH>(args, state,
                                                                 smem);
      return;
    }
  }
  if constexpr (DPT >= 3) {
    if (state.updates_remaining == 2) {
      update_info_shfl<2, COMPUTE_COLS, PROFILE_TYPE, FAST_PATH>(args, state,
                                                                 smem);
      return;
    }
  }
  if constexpr (DPT >= 2) {
    if (state.updates_remaining == 1) {
      update_info_shfl<1, COMPUTE_COLS, PROFILE_TYPE, FAST_PATH>(args, state,
                                                                 smem);
      return;
    }
  }
  if (state.updates_remaining == 0) {
    update_info_shfl<0, COMPUTE_COLS, PROFILE_TYPE, FAST_PATH>(args, state,
                                                               smem);
  }
}

// Per-lane per-row column profile update (in registers; the smem flush
// happens later in update_info_shfl when the slot rotates). Mirrors the
// PROFILE_TYPE dispatch of merge_to_column but operates on the shfl-shaped
// DPT-wide distc array.
template <SCAMPProfileType PROFILE_TYPE, typename T, typename DistType, int DPT,
          typename DerivedDist>
__device__ inline void merge_to_column_shfl(
    const SCAMPKernelInputArgs<double> &args, uint32_t global_row,
    SCAMPShflState<T, DistType, DPT> &state,
    const Eigen::ArrayBase<DerivedDist> &dist) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
#pragma unroll DPT
    for (int i = 0; i < DPT; ++i) {
      if (dist[i] > state.distc[i]) state.distc[i] = dist[i];
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
#pragma unroll DPT
    for (int i = 0; i < DPT; ++i) {
      if (dist[i] > state.distc[i]) {
        state.distc[i] = dist[i];
        state.idxc[i] = global_row;
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    const DistType thresh = static_cast<DistType>(args.opt.threshold);
#pragma unroll DPT
    for (int i = 0; i < DPT; ++i) {
      if (dist[i] > thresh) state.distc[i] += dist[i];
    }
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "merge_to_column_shfl not implemented for profile type");
  }
}

template <SCAMPProfileType PROFILE_TYPE, typename T, typename DistType, int DPT,
          typename DerivedSmem>
__device__ inline void flush_all_cols_to_smem(
    SCAMPShflState<T, DistType, DPT> &state, DerivedSmem &smem) {
#pragma unroll DPT
  for (int i = 0; i < DPT; ++i) {
    if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
      fAtomicMax<ATOMIC_BLOCK>(smem.local_mp_col.data() + state.local_col[i],
                               state.distc[i]);
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                         PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                         PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
      MPatomicMax<ATOMIC_BLOCK>(
          reinterpret_cast<uint64_t *>(smem.local_mp_col.data()) +
              state.local_col[i],
          state.distc[i], state.idxc[i]);
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
      do_atomicAdd<double, ATOMIC_BLOCK>(
          smem.local_mp_col.data() + state.local_col[i],
          static_cast<double>(state.distc[i]));
      state.distc[i] = 0.0;
    }
  }
}

// Per-lane per-row row profile update. Reduces the lane's DPT dist values
// into a single (distr, idxr) for max-style profiles, or a sum for
// SUM_THRESH. Caller handles the warp-reduce + atomic.
template <SCAMPProfileType PROFILE_TYPE, typename DistType, int DPT,
          typename DerivedDist>
__device__ inline void merge_to_row_shfl_lane(
    const SCAMPKernelInputArgs<double> &args,
    const Eigen::Array<uint32_t, DPT, 1> &global_col,
    const Eigen::ArrayBase<DerivedDist> &dist, DistType &distr,
    unsigned int &idxr) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
#pragma unroll DPT
    for (int i = 0; i < DPT; ++i) {
      if (dist[i] > distr) distr = dist[i];
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
#pragma unroll DPT
    for (int i = 0; i < DPT; ++i) {
      if (dist[i] > distr) {
        distr = dist[i];
        idxr = global_col[i];
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    const DistType thresh = static_cast<DistType>(args.opt.threshold);
#pragma unroll DPT
    for (int i = 0; i < DPT; ++i) {
      if (dist[i] > thresh) distr += dist[i];
    }
  }
}

// Warp-wide reduce of (distr, idxr) followed by one atomic to smem from
// lane 0. For max-style profiles this is a max-with-argmax butterfly; for
// SUM_THRESH it's a sum butterfly. One atomic per warp per row.
template <SCAMPProfileType PROFILE_TYPE, typename DistType,
          typename DerivedSmem>
__device__ inline void warp_reduce_and_flush_row(int row_in_tile,
                                                 DistType distr,
                                                 unsigned int idxr,
                                                 uint32_t warpln,
                                                 DerivedSmem &smem) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    DistType sum = distr;
#pragma unroll
    for (int delta = 16; delta >= 1; delta /= 2) {
      sum += __shfl_down_sync(0xffffffffu, sum, delta);
    }
    if (warpln == 0) {
      do_atomicAdd<double, ATOMIC_BLOCK>(smem.local_mp_row.data() + row_in_tile,
                                         static_cast<double>(sum));
    }
  } else {
    // Max-style reduction: track (distr, idxr) as a pair, propagate the
    // winning pair down the warp via __shfl_down_sync.
    DistType d = distr;
    unsigned int x = idxr;
#pragma unroll
    for (int delta = 16; delta >= 1; delta /= 2) {
      DistType d_other = __shfl_down_sync(0xffffffffu, d, delta);
      unsigned int x_other = __shfl_down_sync(0xffffffffu, x, delta);
      if (d_other > d) {
        d = d_other;
        x = x_other;
      }
    }
    if (warpln == 0) {
      if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
        fAtomicMax<ATOMIC_BLOCK>(smem.local_mp_row.data() + row_in_tile, d);
      } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                           PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                           PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
        MPatomicMax<ATOMIC_BLOCK>(
            reinterpret_cast<uint64_t *>(smem.local_mp_row.data()) +
                row_in_tile,
            d, x);
      }
    }
  }
}

// One row of work in the shfl kernel.
//   row_in_tile      — row index within the current tile (0..tile_height-1).
//   tile_start_row   — absolute row of the tile's first row.
//   cov_handoff_smem — pointer to the warp's hand-off slots. Two slots per
//                      warp (double-buffered); the active slot for this row
//                      is selected by `row_in_tile & 1`.
template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          typename DISTANCE_TYPE, int DiagsPerThread, int BLOCKSZ,
          bool FAST_PATH, typename DerivedDataType, typename DerivedSmem>
__device__ inline void do_row_shfl(
    int row_in_tile, uint32_t tile_start_row,
    const SCAMPKernelInputArgs<double> &args,
    SCAMPShflState<DerivedDataType, DISTANCE_TYPE, DiagsPerThread> &state,
    DerivedSmem &smem, DerivedDataType *cov_handoff_smem,
    ShflRowBarrier &row_bar) {
  constexpr int DPT = DiagsPerThread;
  constexpr int warps_per_block = BLOCKSZ / 32;

  const uint32_t global_row =
      tile_start_row + static_cast<uint32_t>(row_in_tile);
  const int read_slot = row_in_tile & 1;
  const int prev_warpid =
      (state.warpid + warps_per_block - 1) % warps_per_block;
  const DerivedDataType cross_warp_in =
      (global_row > 0)
          ? cov_handoff_smem[read_slot * warps_per_block + prev_warpid]
          : DerivedDataType(0);

  if (state.warpln == 0 && global_row > 0) {
    state.cov[0] = cross_warp_in;
  }

  const DerivedDataType dfr = smem.df_row[row_in_tile];
  const DerivedDataType dgr = smem.dg_row[row_in_tile];
  const DerivedDataType inormr = smem.inorm_row[row_in_tile];

  // Compute DPT distances and update cov in place. Mask invalid slots by
  // setting dist to init_dist.
  //
  // EXPERIMENTAL: expressed as Eigen array ops on fixed-size DPT-wide
  // Eigen::Array members. The manual #pragma-unroll'd loop did the same
  // thing scalar-by-scalar; the array form lets Eigen express the same
  // operations as expression templates that nvcc can still fully unroll
  // for small DPT. If perf regresses, fall back to the per-element loop.
  const unsigned int num_diags = args.n_x - args.exclusion_upper + 1;
  const Eigen::Array<DISTANCE_TYPE, DPT, 1> d_raw =
      (state.cov * state.inormc * inormr).template cast<DISTANCE_TYPE>();
  // Cov update uses OLD cov on RHS only -- no aliasing with the d_raw
  // expression above because d_raw was already evaluated into its own
  // storage by the assignment.
  state.cov = state.cov + state.dfc * dgr + state.dgc * dfr;

  // Lane-31 publishes cov[DPT-1] to the next row's read slot. cov[DPT-1]
  // is final after the cov update (above), and nothing in the merges /
  // cov-shift / update_info_shfl that follow depends on cross-warp data.
  // The publish itself is the same write on every arch; only the per-
  // row sync primitive differs (cuda::barrier arrive/wait on sm_70+,
  // __syncthreads on sm_60), so we always publish here.
  {
    const int write_slot = (row_in_tile & 1) ^ 1;
    if (state.warpln == 31) {
      cov_handoff_smem[write_slot * warps_per_block + state.warpid] =
          state.cov[DPT - 1];
    }
  }
  // ARRIVE is non-blocking: on sm_70+ it lets warps that finish the
  // second-half work (~50-150 cycles of merges + shfl + shift) run it
  // concurrently with warps still draining; the WAIT at end of row
  // blocks only on the slowest arrival. On sm_60 cuda::barrier is
  // unavailable and we fall through to a plain __syncthreads() at end
  // of row -- correct, just no latency-hide.
#ifdef SCAMP_SHFL_HAS_CUDA_BARRIER
  auto bar_token = row_bar.arrive();
#else
  (void)row_bar;  // empty stub on sm_60; suppresses unused-arg warning
#endif

  // Slot validity. diag is computed as a signed int array so negative
  // values from the (local_col - row_in_tile) subtraction survive
  // the comparison. The row-out-of-range case in the SLOW PATH short-
  // circuits to all-false because no per-element predicate can recover
  // it (mirrors the bitwise-AND chain in the manual-loop form).
  const Eigen::Array<int, DPT, 1> diag =
      state.local_col.template cast<int>() - row_in_tile;
  constexpr int kMaxDiag = BLOCKSZ * DPT;
  Eigen::Array<bool, DPT, 1> slot_valid = (diag >= 0) && (diag < kMaxDiag);
  if constexpr (!FAST_PATH) {
    if (global_row >= static_cast<uint32_t>(args.n_y)) {
      slot_valid.setZero();
    } else {
      slot_valid = slot_valid &&
                   (state.global_col < static_cast<uint32_t>(args.n_x)) &&
                   ((state.global_col - global_row) < num_diags);
    }
  }
  const Eigen::Array<DISTANCE_TYPE, DPT, 1> dist =
      slot_valid.select(d_raw, init_dist<DISTANCE_TYPE, PROFILE_TYPE>());

  if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    // Per-cell reduction (mirrors CPU update_mp): each lane holds DPT
    // (global_col, dist) pairs at this global_row. Invalid/out-of-bounds slots
    // already carry init_dist (< threshold) so they are skipped by the
    // threshold gate inside ms_accumulate_cell.
#pragma unroll DPT
    for (int i = 0; i < DPT; ++i) {
      ms_accumulate_cell(state, smem, static_cast<int>(global_row),
                         static_cast<int>(state.global_col[i]),
                         static_cast<float>(dist[i]), args);
    }
  } else {
    if constexpr (COMPUTE_COLS) {
      merge_to_column_shfl<PROFILE_TYPE>(args, global_row, state, dist);
    }

    if constexpr (COMPUTE_ROWS) {
      DISTANCE_TYPE distr = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
      unsigned int idxr = 0;
      merge_to_row_shfl_lane<PROFILE_TYPE, DISTANCE_TYPE, DPT>(
          args, state.global_col, dist, distr, idxr);
      warp_reduce_and_flush_row<PROFILE_TYPE>(row_in_tile, distr, idxr,
                                              state.warpln, smem);
    }
  }

  // -----------------------------------------------------------------
  // Cross-warp cov hand-off (double-buffered, one barrier per row).
  //
  // Buffer layout: cov_handoff_smem is 2 * warps_per_block entries.
  //   slot[(row_in_tile & 1)][warpid]      — read here this row
  //   slot[(row_in_tile & 1) ^ 1][warpid]  — write here this row
  //
  // Ordering within a row:
  //   1. Lane 0 of warp k > 0 READS slot[r & 1] (value written by warp k-1
  //      at row r-1; visibility ensured by row r-1's barrier wait).
  //   2. Lane 31 of each warp WRITES slot[(r & 1) ^ 1] (for warp k+1 to
  //      consume at row r+1). Read & write are to DIFFERENT slots, so no
  //      within-row race. Write happens above, right after the cov update.
  //   3. cov shuffle (uses the read value).
  //   4. update_info_shfl + countdown.
  //   5. cuda::barrier WAIT at END of row (paired with the ARRIVE issued
  //      right after the lane-31 publish above).
  //
  // The single arrive/wait pair per row ensures: write-of-row-r
  // happens-before read-of-row-(r+1) (publish), AND read-of-row-r
  // happens-before write-of-row-(r+1) into the same slot (no overwrite).
  // -----------------------------------------------------------------
  const DerivedDataType wrap_in = __shfl_sync(0xffffffffu, state.cov[DPT - 1],
                                              static_cast<int>(state.srcln));

  // Shift cov[i] = cov[i-1] for i in [1, DPT); insert wrap_in at cov[0].
  // .eval() forces a temporary so the tail/head overlap doesn't alias
  // (writing cov[1] would otherwise clobber cov[0]'s value before the
  // next iteration). For DPT <= 8 the temporary is a small register
  // array that nvcc folds into the same registers as the destination.
  if constexpr (DPT > 1) {
    state.cov.template tail<DPT - 1>() =
        state.cov.template head<DPT - 1>().eval();
  }
  state.cov[0] = wrap_in;

  // -----------------------------------------------------------------
  // Column-block rotation (staggered per lane).
  // -----------------------------------------------------------------
  if (state.updates_remaining < DPT) {
    do_update_info_shfl<COMPUTE_COLS, PROFILE_TYPE, FAST_PATH>(args, state,
                                                               smem);
  }
  --state.updates_remaining;

  // End-of-row sync. Orders this row's lane-31 WRITE before next row's
  // lane-0 READ of the same slot (publish), AND this row's lane-0 READ
  // before next row's lane-31 WRITE to the same slot (no overwrite). On
  // sm_70+ this pairs with the cuda::barrier ARRIVE issued earlier in
  // the row; on sm_60 it's a plain __syncthreads() since cuda::barrier
  // isn't available.
#ifdef SCAMP_SHFL_HAS_CUDA_BARRIER
  row_bar.wait(std::move(bar_token));
#else
  __syncthreads();
#endif
}

// -------------------------------------------------------------------------
// SCAMPShflSmem and init helpers
// -------------------------------------------------------------------------
//
// Shared memory layout for the shfl kernel. Differs from SCAMPSmem in two
// ways:
//   1. DROPS df_col / dg_col / inorm_col (column data lives in per-lane
//      registers, refreshed from global on update_info_shfl).
//   2. ADDS cov_handoff (2 * warps_per_block scalars) for the cross-warp
//      cov hand-off. Double-buffered to keep us at one __syncthreads()
//      per row.
//
// Everything else (df_row / dg_row / inorm_row / local_mp_col /
// local_mp_row / profile_a_length / profile_b_length) matches SCAMPSmem,
// so write_back_value / write_back are reused unchanged.
//
// Example smem footprint for a typical shfl variant (DP, DPT=8, BLOCKSZ=128,
// tile_height=256, warps_per_block=4, max-style profile e.g. 1NN_INDEX):
//   df_row + dg_row + inorm_row     = 3 * 256 * 8 = 6144 B
//   local_mp_col                    = 1280 * 8    = 10240 B
//   local_mp_row                    = 256  * 8    =  2048 B
//   cov_handoff                     = 2 * 4 * 8   =    64 B
//   profile_lengths (AAN only)      = 2 * 8       =    16 B
//   ---------------------------------------------------
//   total                                        ≈ 18.5 KB
//
// vs. the sliding-window big-tile variant's ~33 KB at the same tile_height
// (it carries df_col/dg_col/inorm_col in smem too). The shfl variant's
// smaller smem footprint leaves more occupancy headroom on smem-bound
// configurations.

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, SCAMPProfileType type,
          int tile_width, int tile_height, int warps_per_block>
struct SCAMPShflSmem {
  __device__ SCAMPShflSmem(char *smem, bool compute_rows, bool compute_columns,
                           int extra_operands);

  using DataType = DATA_TYPE;
  static constexpr int BLOCKSZ = warps_per_block * 32;

  Eigen::Map<Eigen::Array<DATA_TYPE, tile_height, 1>> df_row;
  Eigen::Map<Eigen::Array<DATA_TYPE, tile_height, 1>> dg_row;
  Eigen::Map<Eigen::Array<DATA_TYPE, tile_height, 1>> inorm_row;
  Eigen::Map<Eigen::Array<PROFILE_DATA_TYPE, tile_width, 1>> local_mp_col;
  Eigen::Map<Eigen::Array<PROFILE_DATA_TYPE, tile_height, 1>> local_mp_row;

  // 2 * warps_per_block entries (double-buffered).
  DATA_TYPE *cov_handoff;

  uint64_t *profile_a_length;
  uint64_t *profile_b_length;

  // MATRIX_SUMMARY per-cell coalescing grid (see SCAMPSmem in
  // kernel_gpu_utils.h). Aliases the local_mp_col region.
  float *ms_grid;
  float *ms_matrix;
  int ms_col_min, ms_row_min, ms_grid_w, ms_grid_h, ms_num_cells;
  bool ms_use_grid;
  bool ms_rowwise;
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, SCAMPProfileType type,
          int tile_width, int tile_height, int warps_per_block>
__device__
SCAMPShflSmem<DATA_TYPE, PROFILE_DATA_TYPE, type, tile_width, tile_height,
              warps_per_block>::SCAMPShflSmem(char *smem, bool compute_rows,
                                              bool compute_columns,
                                              int extra_operands)
    : df_row(nullptr),
      dg_row(nullptr),
      inorm_row(nullptr),
      local_mp_col(nullptr),
      local_mp_row(nullptr) {
  using TallArrayMap = decltype(df_row);
  using ColProfileMap = decltype(local_mp_col);
  using RowProfileMap = decltype(local_mp_row);

  new (&df_row) TallArrayMap(reinterpret_cast<DATA_TYPE *>(smem));
  smem += sizeof(DATA_TYPE) * tile_height;
  new (&dg_row) TallArrayMap(reinterpret_cast<DATA_TYPE *>(smem));
  smem += sizeof(DATA_TYPE) * tile_height;
  new (&inorm_row) TallArrayMap(reinterpret_cast<DATA_TYPE *>(smem));
  smem += sizeof(DATA_TYPE) * tile_height;

  if (compute_columns) {
    new (&local_mp_col)
        ColProfileMap(reinterpret_cast<PROFILE_DATA_TYPE *>(smem));
    smem += sizeof(PROFILE_DATA_TYPE) * tile_width;
  }
  if (compute_rows) {
    new (&local_mp_row)
        RowProfileMap(reinterpret_cast<PROFILE_DATA_TYPE *>(smem));
    smem += sizeof(PROFILE_DATA_TYPE) * tile_height;
  }

  cov_handoff = reinterpret_cast<DATA_TYPE *>(smem);
  smem += sizeof(DATA_TYPE) * 2 * warps_per_block;

  if (NeedsCheckIfDone(type)) {
    profile_a_length = reinterpret_cast<uint64_t *>(smem);
    smem += sizeof(uint64_t);
    profile_b_length = reinterpret_cast<uint64_t *>(smem);
  } else {
    profile_a_length = nullptr;
    profile_b_length = nullptr;
  }
  ms_grid = compute_columns
                ? reinterpret_cast<float *>(local_mp_col.data())
                : (compute_rows ? reinterpret_cast<float *>(local_mp_row.data())
                                : nullptr);
  ms_matrix = nullptr;
  ms_col_min = ms_row_min = 0;
  ms_grid_w = ms_grid_h = ms_num_cells = 0;
  ms_use_grid = false;
  ms_rowwise = false;
}

// init_smem_shfl: parallels the four init_smem variants in kernels_smem.h
// but skips the df_col/dg_col/inorm_col loads (those regions don't exist
// in SCAMPShflSmem). Profile-type dispatch matches the existing one.

template <typename SMEM_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ>
__device__ inline void init_smem_shfl_with_static_initializer(
    SCAMPKernelInputArgs<double> &args, SMEM_TYPE &smem, uint32_t col_start,
    uint32_t row_start, PROFILE_DATA_TYPE initializer) {
  int global_position = col_start + threadIdx.x;
  int local_position = threadIdx.x;
  while (local_position < tile_width && global_position < args.n_x) {
    if (COMPUTE_COLS) {
      smem.local_mp_col[local_position] = initializer;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }

  global_position = row_start + threadIdx.x;
  local_position = threadIdx.x;
  while (local_position < tile_height && global_position < args.n_y) {
    smem.dg_row[local_position] = args.dgb[global_position];
    smem.df_row[local_position] = args.dfb[global_position];
    smem.inorm_row[local_position] = args.normsb[global_position];
    if (COMPUTE_ROWS) {
      smem.local_mp_row[local_position] = initializer;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }
}

template <typename SMEM_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ>
__device__ inline void init_smem_shfl_with_dynamic_initializer(
    SCAMPKernelInputArgs<double> &args, SMEM_TYPE &smem,
    PROFILE_DATA_TYPE *initializer_col, PROFILE_DATA_TYPE *initializer_row,
    uint32_t col_start, uint32_t row_start) {
  int global_position = col_start + threadIdx.x;
  int local_position = threadIdx.x;
  while (local_position < tile_width && global_position < args.n_x) {
    if (COMPUTE_COLS) {
      smem.local_mp_col[local_position] = initializer_col[global_position];
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }

  global_position = row_start + threadIdx.x;
  local_position = threadIdx.x;
  while (local_position < tile_height && global_position < args.n_y) {
    smem.dg_row[local_position] = args.dgb[global_position];
    smem.df_row[local_position] = args.dfb[global_position];
    smem.inorm_row[local_position] = args.normsb[global_position];
    if (COMPUTE_ROWS) {
      smem.local_mp_row[local_position] = initializer_row[global_position];
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }
}

template <typename SMEM_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ>
__device__ inline void init_smem_shfl_for_all_neighbors(
    SCAMPKernelInputArgs<double> &args, SMEM_TYPE &smem, uint32_t col_start,
    uint32_t row_start) {
  int global_position = col_start + threadIdx.x;
  int local_position = threadIdx.x;
  mp_entry initializer;
  initializer.ints[1] = 0;
  while (local_position < tile_width && global_position < args.n_x) {
    if (COMPUTE_COLS) {
      initializer.floats[0] = args.thresholds_a[global_position];
      smem.local_mp_col[local_position] = initializer.ulong;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }

  global_position = row_start + threadIdx.x;
  local_position = threadIdx.x;
  while (local_position < tile_height && global_position < args.n_y) {
    smem.dg_row[local_position] = args.dgb[global_position];
    smem.df_row[local_position] = args.dfb[global_position];
    smem.inorm_row[local_position] = args.normsb[global_position];
    if (COMPUTE_ROWS) {
      initializer.floats[0] = args.thresholds_b[global_position];
      smem.local_mp_row[local_position] = initializer.ulong;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }
}

template <typename SMEM_TYPE, typename PROFILE_DATA_TYPE,
          typename PROFILE_OUTPUT_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
__device__ void init_smem_shfl(SCAMPKernelInputArgs<double> &args,
                               SMEM_TYPE &smem, PROFILE_OUTPUT_TYPE *profile_a,
                               PROFILE_OUTPUT_TYPE *profile_b,
                               uint32_t col_start, uint32_t row_start) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                PROFILE_TYPE == PROFILE_TYPE_1NN) {
    init_smem_shfl_with_dynamic_initializer<SMEM_TYPE, PROFILE_DATA_TYPE,
                                            COMPUTE_ROWS, COMPUTE_COLS,
                                            tile_width, tile_height, BLOCKSZ>(
        args, smem, profile_a, profile_b, col_start, row_start);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    init_smem_shfl_with_static_initializer<SMEM_TYPE, PROFILE_DATA_TYPE,
                                           COMPUTE_ROWS, COMPUTE_COLS,
                                           tile_width, tile_height, BLOCKSZ>(
        args, smem, col_start, row_start, 0.0);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    // Load the df/dg/inorm row data into smem but skip the per-column/row
    // profile init: matrix summary uses the cell grid below instead.
    init_smem_shfl_with_static_initializer<SMEM_TYPE, PROFILE_DATA_TYPE,
                                           /*COMPUTE_ROWS=*/false,
                                           /*COMPUTE_COLS=*/false, tile_width,
                                           tile_height, BLOCKSZ>(
        args, smem, col_start, row_start, static_cast<PROFILE_DATA_TYPE>(0));
    // Orientation + cell-grid setup (see the sliding-window init_smem for the
    // rationale). COMPUTE_ROWS marks the transposed run, which writes profile_b
    // (the real matrix in that run) with row/col swapped.
    constexpr bool kRowwise = COMPUTE_ROWS;
    smem.ms_rowwise = kRowwise;
    smem.ms_matrix =
        reinterpret_cast<float *>(kRowwise ? profile_b : profile_a);
    int rb0, cb0, rb1, cb1;
    ms_cell_of(static_cast<int>(row_start), static_cast<int>(col_start),
               kRowwise, args, &rb0, &cb0);
    ms_cell_of(static_cast<int>(row_start + tile_height - 1),
               static_cast<int>(col_start + tile_width + tile_height - 1),
               kRowwise, args, &rb1, &cb1);
    smem.ms_row_min = rb0 < rb1 ? rb0 : rb1;
    smem.ms_col_min = cb0 < cb1 ? cb0 : cb1;
    smem.ms_grid_h = (rb0 > rb1 ? rb0 : rb1) - smem.ms_row_min + 1;
    smem.ms_grid_w = (cb0 > cb1 ? cb0 : cb1) - smem.ms_col_min + 1;
    smem.ms_num_cells = smem.ms_grid_w * smem.ms_grid_h;
    // Cap = capacity of the aliased region (local_mp_col on the normal run,
    // local_mp_row on the transposed run).
    constexpr int kMsGridCap =
        static_cast<int>(sizeof(PROFILE_DATA_TYPE) / sizeof(float)) *
        (kRowwise ? tile_height : tile_width);
    smem.ms_use_grid = smem.ms_num_cells > 0 && smem.ms_num_cells <= kMsGridCap;
    if (smem.ms_use_grid) {
      for (int idx = threadIdx.x; idx < smem.ms_num_cells; idx += BLOCKSZ) {
        smem.ms_grid[idx] = -2.0f;
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    init_smem_shfl_for_all_neighbors<SMEM_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                                     tile_width, tile_height, BLOCKSZ>(
        args, smem, col_start, row_start);
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "init_smem_shfl not implemented for profile type.");
  }
}

}  // namespace SCAMP
