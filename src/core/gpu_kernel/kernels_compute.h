#pragma once

#include "core/defines.h"

//////////////////////////////////////////////////////
// Helpers: NaN-safe array reductions for the profile-row update.
//
// Eigen's maxCoeff<PropagateNumbers>() is the natural choice when the
// array is large enough that the visitor's overhead is amortized; for
// the tiny <=4-element inner-loop arrays we hand-roll the scan to keep
// register pressure low. The compile-time-N branch picks the right
// strategy.
//////////////////////////////////////////////////////

template <typename Derived, typename ScalarType = typename Derived::Scalar>
__device__ inline ScalarType max_dist(const Eigen::ArrayBase<Derived> &dist) {
  ScalarType ret = -2;
  if constexpr (Derived::RowsAtCompileTime > 4) {
    ScalarType max = dist.template maxCoeff<Eigen::PropagateNumbers>();
    if (max > ret) {
      ret = max;
    }
    return ret;
  }
  for_<Derived::RowsAtCompileTime>([&](auto i) {
    if (dist[i.value] > ret) {
      ret = dist[i.value];
    }
  });
  return ret;
}

template <typename Derived, typename ScalarType = typename Derived::Scalar>
__device__ inline ScalarType max_dist(const Eigen::ArrayBase<Derived> &dist,
                                      int &idx) {
  ScalarType ret = -2;
  if constexpr (Derived::RowsAtCompileTime > 4) {
    ScalarType max = dist.template maxCoeff<Eigen::PropagateNumbers>(&idx);
    if (max > ret) {
      ret = max;
    }
    return ret;
  }
  for_<Derived::RowsAtCompileTime>([&](auto i) {
    if (dist[i.value] > ret) {
      ret = dist[i.value];
      idx = i.value;
    }
  });
  return ret;
}

//////////////////////////////////////////////////////
// MERGE_TO_ROW: per-iter row update.
//
// For each "thread row" of DiagsPerThread distances (one row's worth of
// the unrolled parallelogram), reduce into the running per-thread row
// best (distr/idxr). The profile-type-specific behavior:
//   1NN:                       max-ignoring-nan
//   1NN_INDEX / MATRIX_SUMMARY /
//     APPROX_ALL_NEIGHBORS:    max + index
//   SUM_THRESH:                threshold-gated sum
//////////////////////////////////////////////////////

template <int row_iter, SCAMPProfileType PROFILE_TYPE, typename DISTANCE_TYPE,
          typename InputDataType, int DiagsPerThread, typename DerivedSmem,
          typename DistRowArray>
__device__ inline void merge_to_row(
    const SCAMPKernelInputArgs<double> &args,
    const SCAMPThreadInfo<InputDataType, DiagsPerThread> &info,
    DerivedSmem &smem, const Eigen::ArrayBase<DistRowArray> &dist,
    DISTANCE_TYPE &distr, unsigned int &idxr) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    DISTANCE_TYPE d = max_dist(dist);
    distr = fmaxf(distr, d);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    int idx = 0;
    DISTANCE_TYPE d = max_dist(dist, idx);
    idx += info.global_col + row_iter;
    if (d > distr) {
      distr = d;
      idxr = idx;
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    DISTANCE_TYPE sum = (dist > args.opt.threshold).select(dist, 0).sum();
    distr += sum;
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "merge_to_row not implemented for profile type.");
  }
}

//////////////////////////////////////////////////////
// UPDATE_ROWS: flush a batch of per-thread row bests to smem.
//
// Called after each inner-loop batch of `num_to_update` rows. For most
// profile types this is num_to_update test-and-test-and-set atomics
// against shared memory; for SUM_THRESH it's a warp-shuffle reduction
// followed by one atomicAdd per warp.
//////////////////////////////////////////////////////

template <int row_iter, int num_to_update, SCAMPProfileType PROFILE_TYPE,
          typename DISTANCE_TYPE, typename InputDataType, int DiagsPerThread,
          typename DerivedSmem, typename DistRowArray, typename IndexRowArray>
__device__ inline void update_rows(
    const SCAMPKernelInputArgs<double> &args,
    const SCAMPThreadInfo<InputDataType, DiagsPerThread> &info,
    DerivedSmem &smem, const Eigen::ArrayBase<DistRowArray> &distr,
    const Eigen::ArrayBase<IndexRowArray> &idxr) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    Eigen::Array<float, num_to_update, 1> mp_row_check =
        smem.local_mp_row.template segment<num_to_update>(info.local_row +
                                                          row_iter);
#pragma unroll num_to_update
    for (int i = 0; i < num_to_update; ++i) {
      fAtomicMax_check<ATOMIC_BLOCK>(
          smem.local_mp_row.data() + info.local_row + i + row_iter,
          distr[i + row_iter], mp_row_check[i]);
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    Eigen::Array<float, num_to_update, 1> mp_row_check;
    Eigen::Array<uint64_t, num_to_update, 1> temp =
        smem.local_mp_row.template segment<num_to_update>(info.local_row +
                                                          row_iter);
#pragma unroll num_to_update
    for (int i = 0; i < num_to_update; ++i) {
      mp_entry e;
      e.ulong = temp[i];
      mp_row_check[i] = e.floats[0];
    }
#pragma unroll num_to_update
    for (int r = 0; r < num_to_update; ++r) {
      MPatomicMax_check<ATOMIC_BLOCK>(
          reinterpret_cast<uint64_t *>(smem.local_mp_row.data()) +
              info.local_row + r + row_iter,
          distr[r + row_iter], idxr[r + row_iter], mp_row_check[r]);
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
#pragma unroll num_to_update
    for (int r = 0; r < num_to_update; ++r) {
      DISTANCE_TYPE sum = distr[r + row_iter];
#pragma unroll
      for (int i = 16; i >= 1; i /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, i);
      }
      if ((threadIdx.x & 0x1f) == 0) {
        do_atomicAdd<double, ATOMIC_BLOCK>(
            smem.local_mp_row.data() + info.local_row + r + row_iter,
            static_cast<double>(sum));
      }
    }
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "update_rows not implemented for profile type.");
  }
}

//////////////////////////////////////////////////////
// MERGE_TO_COLUMN: per-iter column best-so-far update.
//
// For each row of DiagsPerThread distances, merge into the running
// per-thread per-column best (best_so_far / best_so_far_index). For
// SUM_THRESH this is a vectorized threshold-gated sum; for the others
// it's a pairwise compare-and-swap.
//////////////////////////////////////////////////////

template <int row_iter, SCAMPProfileType PROFILE_TYPE, int DiagsPerThread,
          typename DerivedDataType, typename DerivedSmem, typename ColDistArray,
          typename RowDistArray, typename ColIndexArray>
__device__ inline void merge_to_column(
    const SCAMPKernelInputArgs<double> &args,
    const SCAMPThreadInfo<DerivedDataType, DiagsPerThread> &info,
    DerivedSmem smem, Eigen::ArrayBase<ColDistArray> &best_so_far,
    const Eigen::ArrayBase<RowDistArray> &dists_to_merge,
    Eigen::ArrayBase<ColIndexArray> &best_so_far_index) {
  constexpr int unrolled_diags = DiagsPerThread;
  static_assert(RowDistArray::RowsAtCompileTime == unrolled_diags,
                "dists_to_merge must have DiagsPerThread rows.");
  static_assert(ColDistArray::RowsAtCompileTime ==
                    ColIndexArray::RowsAtCompileTime,
                "best_so_far and best_so_far_index sizes must match.");
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
#pragma unroll unrolled_diags
    for (int i = 0; i < unrolled_diags; ++i) {
      if (dists_to_merge[i] > best_so_far[row_iter + i]) {
        best_so_far[row_iter + i] = dists_to_merge[i];
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
#pragma unroll unrolled_diags
    for (int i = 0; i < unrolled_diags; ++i) {
      if (dists_to_merge[i] > best_so_far[row_iter + i]) {
        best_so_far[row_iter + i] = dists_to_merge[i];
        best_so_far_index[row_iter + i] = info.global_row + row_iter;
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    best_so_far.template segment<unrolled_diags>(row_iter) +=
        (dists_to_merge > args.opt.threshold).select(dists_to_merge, 0);
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "merge_to_column not implemented for profile type.");
  }
}

//////////////////////////////////////////////////////
// UPDATE_COLS: flush a batch of per-thread col bests to smem.
//
// Mirrors update_rows but writes to local_mp_col. Called from
// do_iteration_fast after each row-batch to coalesce per-thread column
// bests into the block's shared profile.
//////////////////////////////////////////////////////

template <int start_index, int num_to_update, SCAMPProfileType PROFILE_TYPE,
          typename DerivedInputDataType, int DiagsPerThread,
          typename DerivedSmemType, typename ColDistArray,
          typename ColIndexArray>
__device__ inline void update_cols(
    const SCAMPKernelInputArgs<double> &args,
    SCAMPThreadInfo<DerivedInputDataType, DiagsPerThread> &info,
    DerivedSmemType &smem, Eigen::ArrayBase<ColDistArray> &distc,
    Eigen::ArrayBase<ColIndexArray> &idxc) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    Eigen::Array<float, num_to_update, 1> mp_col_check =
        smem.local_mp_col.template segment<num_to_update>(info.local_col +
                                                          start_index);
#pragma unroll num_to_update
    for (int i = 0; i < num_to_update; ++i) {
      fAtomicMax_check<ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + i + start_index,
          distc[i + start_index], mp_col_check[i]);
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    Eigen::Array<float, num_to_update, 1> mp_col_check;
    {
      Eigen::Array<uint64_t, num_to_update, 1> temp =
          smem.local_mp_col.template segment<num_to_update>(info.local_col +
                                                            start_index);
#pragma unroll num_to_update
      for (int i = 0; i < num_to_update; ++i) {
        mp_entry e;
        e.ulong = temp[i];
        mp_col_check[i] = e.floats[0];
      }
    }
#pragma unroll num_to_update
    for (int i = 0; i < num_to_update; ++i) {
      MPatomicMax_check<ATOMIC_BLOCK>(
          reinterpret_cast<uint64_t *>(smem.local_mp_col.data()) +
              info.local_col + i + start_index,
          distc[i + start_index], idxc[i + start_index], mp_col_check[i]);
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
#pragma unroll num_to_update
    for (int i = 0; i < num_to_update; ++i) {
      do_atomicAdd<double, ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + i + start_index,
          distc[i + start_index]);
    }
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "update_cols not implemented for profile type.");
  }
}

/////////////////////////////////////////////////////
// DO_ROW: one row of DiagsPerThread distances.
//
// outer_row_iter: absolute row index within the outer parallelogram
//   (drives the per-column best_so_far offset). Range
//   0..OuterUnrolledRows.
// row_iter: the inner-loop k index (drives the per-column smem
//   register-window offset). Range 0..UnrolledRows.
//
// info.cov, inormc, dfc, dgc are arrays held in registers; inormr, dfr,
// dgr are scalars (the caller indexes the row arrays).
/////////////////////////////////////////////////////
template <int outer_row_iter, int row_iter, SCAMPProfileType PROFILE_TYPE,
          bool COMPUTE_ROWS, bool COMPUTE_COLS, typename DISTANCE_TYPE,
          typename DerivedInputType, int DiagsPerThread, typename DerivedSmem,
          typename DistColArray, typename InputColArray, typename IndexColArray>
__device__ inline FORCE_INLINE void do_row(
    const SCAMPKernelInputArgs<double> &args,
    SCAMPThreadInfo<DerivedInputType, DiagsPerThread> &info, DerivedSmem &smem,
    Eigen::ArrayBase<DistColArray> &distc, DISTANCE_TYPE &distr,
    const Eigen::ArrayBase<InputColArray> &inormc,
    const Eigen::ArrayBase<InputColArray> &dfc,
    const Eigen::ArrayBase<InputColArray> &dgc,
    const typename InputColArray::Scalar inormr,
    const typename InputColArray::Scalar dfr,
    const typename InputColArray::Scalar dgr,
    Eigen::ArrayBase<IndexColArray> &idxc, unsigned int &idxr) {
  constexpr int unrolled_diags = DiagsPerThread;
  Eigen::Array<DISTANCE_TYPE, unrolled_diags, 1> dist;
#pragma unroll unrolled_diags
  for (int i = 0; i < unrolled_diags; ++i) {
    dist[i] = static_cast<DISTANCE_TYPE>(info.cov[i] * inormc[row_iter + i] *
                                         inormr);
    info.cov[i] =
        info.cov[i] + dfc[row_iter + i] * dgr + dgc[row_iter + i] * dfr;
  }
  if constexpr (COMPUTE_COLS) {
    merge_to_column<outer_row_iter, PROFILE_TYPE, DiagsPerThread>(
        args, info, smem, distc, dist, idxc);
  }
  if constexpr (COMPUTE_ROWS) {
    merge_to_row<outer_row_iter, PROFILE_TYPE, DISTANCE_TYPE, DerivedInputType,
                 DiagsPerThread>(args, info, smem, dist, distr, idxr);
  }
}

///////////////////////////////////////////////////////////////////////////////
// OPTIMIZED CODE PATH:
// do_iteration_fast processes OuterUnrolledRows of work per call. It uses a
// register-resident column-data sliding window of width
// inner_unrolled_cols = DiagsPerThread + UnrolledRows - 1, refilling from
// smem after each inner row-batch of size UnrolledRows.
//
// Per call:
//   - per-thread work: OuterUnrolledRows rows x DiagsPerThread diagonals
//     = OuterUnrolledRows * DiagsPerThread distances
//   - smem column reads: inner_unrolled_cols on entry, then
//     UnrolledRows per inner iteration (after the first)
///////////////////////////////////////////////////////////////////////////////
template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          typename DISTANCE_TYPE, int DiagsPerThread, int UnrolledRows,
          int OuterUnrolledRows, typename DerivedDataType, typename DerivedSmem>
__device__ void do_iteration_fast(
    const SCAMPKernelInputArgs<double> &args,
    SCAMPThreadInfo<DerivedDataType, DiagsPerThread> &info, DerivedSmem &smem) {
  constexpr int inner_unrolled_cols = DiagsPerThread + UnrolledRows - 1;
  constexpr int unrolled_cols = DiagsPerThread + OuterUnrolledRows - 1;

  // Local register-window arrays use the smem column-data type. Since
  // MIXED was dropped, DataType always equals the cov accumulator type,
  // but routing through DerivedSmem::DataType keeps the do_iteration_fast
  // body decoupled from how the SCAMPThreadInfo template args got chosen.
  using SmemDataType = typename DerivedSmem::DataType;
  Eigen::Array<SmemDataType, inner_unrolled_cols, 1> dfc, dgc, inormc;
  DISTANCE_TYPE init = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  Eigen::Array<DISTANCE_TYPE, unrolled_cols, 1> distc =
      Eigen::Array<DISTANCE_TYPE, unrolled_cols, 1>::Constant(init);
  Eigen::Array<DISTANCE_TYPE, OuterUnrolledRows, 1> distr =
      Eigen::Array<DISTANCE_TYPE, OuterUnrolledRows, 1>::Constant(init);
  Eigen::Array<unsigned int, unrolled_cols, 1> idxc;
  Eigen::Array<unsigned int, OuterUnrolledRows, 1> idxr;

  // Initial load of the column sliding-window from smem.
  dfc = smem.df_col.template segment<inner_unrolled_cols>(info.local_col);
  dgc = smem.dg_col.template segment<inner_unrolled_cols>(info.local_col);
  inormc =
      smem.inorm_col.template segment<inner_unrolled_cols>(info.local_col);

  // Outer loop: process OuterUnrolledRows rows in batches of UnrolledRows.
  for_<OuterUnrolledRows / UnrolledRows>([&](auto j) {
    if constexpr (j.value > 0) {
      // Slide the column window left by UnrolledRows and load the next
      // UnrolledRows columns into the right edge.
      dfc.template segment<inner_unrolled_cols - UnrolledRows>(0) =
          dfc.template segment<inner_unrolled_cols - UnrolledRows>(
              UnrolledRows);
      dgc.template segment<inner_unrolled_cols - UnrolledRows>(0) =
          dgc.template segment<inner_unrolled_cols - UnrolledRows>(
              UnrolledRows);
      inormc.template segment<inner_unrolled_cols - UnrolledRows>(0) =
          inormc.template segment<inner_unrolled_cols - UnrolledRows>(
              UnrolledRows);
      dfc.template segment<UnrolledRows>(inner_unrolled_cols - UnrolledRows) =
          smem.df_col.template segment<UnrolledRows>(
              info.local_col + j.value * UnrolledRows +
              (inner_unrolled_cols - UnrolledRows));
      dgc.template segment<UnrolledRows>(inner_unrolled_cols - UnrolledRows) =
          smem.dg_col.template segment<UnrolledRows>(
              info.local_col + j.value * UnrolledRows +
              (inner_unrolled_cols - UnrolledRows));
      inormc.template segment<UnrolledRows>(inner_unrolled_cols -
                                            UnrolledRows) =
          smem.inorm_col.template segment<UnrolledRows>(
              info.local_col + j.value * UnrolledRows +
              (inner_unrolled_cols - UnrolledRows));
    }
    Eigen::Array<SmemDataType, UnrolledRows, 1> dfr =
        smem.df_row.template segment<UnrolledRows>(info.local_row +
                                                   j.value * UnrolledRows);
    Eigen::Array<SmemDataType, UnrolledRows, 1> dgr =
        smem.dg_row.template segment<UnrolledRows>(info.local_row +
                                                   j.value * UnrolledRows);
    Eigen::Array<SmemDataType, UnrolledRows, 1> inormr =
        smem.inorm_row.template segment<UnrolledRows>(info.local_row +
                                                      j.value * UnrolledRows);
    for_<UnrolledRows>([&](auto k) {
      do_row<j.value * UnrolledRows + k.value, k.value, PROFILE_TYPE,
             COMPUTE_ROWS, COMPUTE_COLS, DISTANCE_TYPE, DerivedDataType,
             DiagsPerThread>(args, info, smem, distc,
                             distr[j.value * UnrolledRows + k.value], inormc,
                             dfc, dgc, inormr[k.value], dfr[k.value],
                             dgr[k.value], idxc,
                             idxr[j.value * UnrolledRows + k.value]);
    });
    if constexpr (COMPUTE_COLS) {
      update_cols<j.value * UnrolledRows, UnrolledRows, PROFILE_TYPE,
                  DerivedDataType, DiagsPerThread>(args, info, smem, distc,
                                                   idxc);
    }
    if constexpr (COMPUTE_ROWS) {
      update_rows<j.value * UnrolledRows, UnrolledRows, PROFILE_TYPE,
                  DISTANCE_TYPE, DerivedDataType, DiagsPerThread>(
          args, info, smem, distr, idxr);
    }
  });

  // Flush the trailing tail of distc (positions OuterUnrolledRows
  // through unrolled_cols-1 hold bests merged but never atomic-flushed
  // by the per-batch update above).
  if constexpr (COMPUTE_COLS) {
    update_cols<OuterUnrolledRows, unrolled_cols - OuterUnrolledRows,
                PROFILE_TYPE, DerivedDataType, DiagsPerThread>(args, info, smem,
                                                               distc, idxc);
  }
  info.local_col += OuterUnrolledRows;
  info.local_row += OuterUnrolledRows;
  info.global_col += OuterUnrolledRows;
  info.global_row += OuterUnrolledRows;
}

/////////////////////////////////////////////////////////////////////////
//  EDGE COMPUTATION
//
// reduce_row/reduce_edge/do_row_edge are the slow-path equivalents of
// merge_to_row/merge_to_column/do_row, called one row at a time near
// tile boundaries where the safe-to-read window does not span
// DiagsPerThread columns.
//////////////////////////////////////////////////////////////////////

template <SCAMPProfileType PROFILE_TYPE, typename DerivedDataType,
          int DiagsPerThread, typename DerivedDist, typename DerivedSmemType>
__device__ inline void reduce_row(
    const SCAMPKernelInputArgs<double> &args,
    const SCAMPThreadInfo<DerivedDataType, DiagsPerThread> &info,
    DerivedSmemType &smem, DerivedDist dist_row, uint32_t idx_row) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_row.data() + info.local_row,
                                   dist_row, -2);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    MPatomicMax_check<ATOMIC_BLOCK>(
        reinterpret_cast<uint64_t *>(smem.local_mp_row.data()) +
            info.local_row,
        dist_row, idx_row, -2);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    do_atomicAdd<double, ATOMIC_BLOCK>(
        smem.local_mp_row.data() + info.local_row, dist_row);
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "reduce_row not implemented for profile type.");
  }
}

template <int iter, SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, typename DerivedSmemType, typename DerivedDataType,
          int DiagsPerThread, typename DerivedDist, typename DerivedDist4>
__device__ inline void reduce_edge(
    const SCAMPKernelInputArgs<double> &args,
    const SCAMPThreadInfo<DerivedDataType, DiagsPerThread> &info,
    DerivedSmemType &smem, const Eigen::ArrayBase<DerivedDist4> &dist,
    DerivedDist &dist_row, uint32_t &idx_row, int diag, int num_diags) {
  if (info.global_col + iter < args.n_x && diag + iter < num_diags) {
    if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
      if (!isnan(dist[iter])) {
        if constexpr (COMPUTE_ROWS) {
          dist_row = fmaxf(dist_row, dist[iter]);
        }
        if constexpr (COMPUTE_COLS) {
          fAtomicMax<ATOMIC_BLOCK>(smem.local_mp_col.data() + info.local_col +
                                       iter,
                                   dist[iter]);
        }
      }
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                         PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                         PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
      if constexpr (COMPUTE_ROWS) {
        if (dist[iter] > dist_row) {
          dist_row = dist[iter];
          idx_row = info.global_col + iter;
        }
      }
      if constexpr (COMPUTE_COLS) {
        MPatomicMax<ATOMIC_BLOCK>(
            reinterpret_cast<uint64_t *>(smem.local_mp_col.data()) +
                info.local_col + iter,
            dist[iter], info.global_row);
      }
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
      if (dist[iter] > args.opt.threshold) {
        if constexpr (COMPUTE_ROWS) {
          dist_row += dist[iter];
        }
        if constexpr (COMPUTE_COLS) {
          do_atomicAdd<double, ATOMIC_BLOCK>(
              smem.local_mp_col.data() + info.local_col + iter, dist[iter]);
        }
      }
    } else {
      static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                    "reduce_edge not implemented for profile type.");
    }
  }
}

template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          typename DISTANCE_TYPE, int DiagsPerThread, typename DerivedSmemType,
          typename DerivedDataType>
__device__ inline void do_row_edge(
    const SCAMPKernelInputArgs<double> &args,
    SCAMPThreadInfo<DerivedDataType, DiagsPerThread> &info,
    DerivedSmemType &smem, int diag, int num_diags) {
  DISTANCE_TYPE dist_row = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  uint32_t idx_row = 0;
  DerivedDataType inormr = smem.inorm_row[info.local_row];
  DerivedDataType dgr = smem.dg_row[info.local_row];
  DerivedDataType dfr = smem.df_row[info.local_row];

  // Compute DiagsPerThread distances at the current edge position. Some
  // entries may correspond to out-of-bounds positions; the reduce_edge
  // calls below bound-check before consuming each one.
  //
  // Element-wise scalar form (not an Eigen array expression) keeps the
  // same shape do_iteration_fast uses. Originally needed because
  // PRECISION_MIXED could put info.cov (double) and the smem segments
  // (float) at different scalar types under Eigen 5's strict promotion;
  // now that MIXED is dropped both are the same type, but the scalar form
  // costs nothing and is consistent with do_row.
  Eigen::Array<DISTANCE_TYPE, DiagsPerThread, 1> dist;
#pragma unroll DiagsPerThread
  for (int i = 0; i < DiagsPerThread; ++i) {
    dist[i] =
        static_cast<DISTANCE_TYPE>(info.cov[i] *
                                   smem.inorm_col[info.local_col + i] * inormr);
    info.cov[i] = info.cov[i] + smem.df_col[info.local_col + i] * dgr +
                  smem.dg_col[info.local_col + i] * dfr;
  }

  for_<DiagsPerThread>([&](auto i) {
    reduce_edge<i.value, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                DerivedSmemType, DerivedDataType, DiagsPerThread>(
        args, info, smem, dist, dist_row, idx_row, diag, num_diags);
  });

  if constexpr (COMPUTE_ROWS) {
    reduce_row<PROFILE_TYPE, DerivedDataType, DiagsPerThread>(
        args, info, smem, dist_row, idx_row);
  }
}
