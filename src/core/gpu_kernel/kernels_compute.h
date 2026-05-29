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
//
// row_iter is a runtime int (offset into the outer parallelogram). It is
// only used as an offset / addend (no template-arg consumers), so passing
// it as a function argument lets the caller drop its per-iteration
// template specialization and emit a single inlined body. See
// do_iteration_fast's #pragma-unrolled outer loop.
//////////////////////////////////////////////////////

template <SCAMPProfileType PROFILE_TYPE, typename DISTANCE_TYPE,
          typename InputDataType, int DiagsPerThread, typename DerivedSmem,
          typename DistRowArray>
__device__ inline void merge_to_row(
    int row_iter, const SCAMPKernelInputArgs<double> &args,
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
//
// row_iter is runtime; num_to_update stays compile-time (Eigen::Array
// size + #pragma unroll bound).
//////////////////////////////////////////////////////

template <int num_to_update, SCAMPProfileType PROFILE_TYPE,
          typename DISTANCE_TYPE, typename InputDataType, int DiagsPerThread,
          typename DerivedSmem, typename DistRowArray, typename IndexRowArray>
__device__ inline void update_rows(
    int row_iter, const SCAMPKernelInputArgs<double> &args,
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
//
// row_iter is runtime.
//////////////////////////////////////////////////////

template <SCAMPProfileType PROFILE_TYPE, int DiagsPerThread,
          typename DerivedDataType, typename DerivedSmem, typename ColDistArray,
          typename RowDistArray, typename ColIndexArray>
__device__ inline void merge_to_column(
    int row_iter, const SCAMPKernelInputArgs<double> &args,
    const SCAMPThreadInfo<DerivedDataType, DiagsPerThread> &info,
    DerivedSmem smem, Eigen::ArrayBase<ColDistArray> &best_so_far,
    const Eigen::ArrayBase<RowDistArray> &dists_to_merge,
    Eigen::ArrayBase<ColIndexArray> &best_so_far_index) {
  constexpr int unrolled_diags = DiagsPerThread;
  static_assert(RowDistArray::RowsAtCompileTime == unrolled_diags,
                "dists_to_merge must have DiagsPerThread rows.");
  static_assert(
      ColDistArray::RowsAtCompileTime == ColIndexArray::RowsAtCompileTime,
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
//
// start_index is runtime; num_to_update stays compile-time.
//////////////////////////////////////////////////////

template <int num_to_update, SCAMPProfileType PROFILE_TYPE,
          typename DerivedInputDataType, int DiagsPerThread,
          typename DerivedSmemType, typename ColDistArray,
          typename ColIndexArray>
__device__ inline void update_cols(
    int start_index, const SCAMPKernelInputArgs<double> &args,
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
// outer_row_iter (runtime): absolute row index within the outer
//   parallelogram (drives the per-column best_so_far offset). Range
//   0..OuterUnrolledRows.
// row_iter (runtime): the inner-loop k index (drives the per-column smem
//   register-window offset). Range 0..UnrolledRows.
//
// Both are runtime to avoid one do_row template instantiation per
// (outer_row_iter, row_iter) pair; the caller emits a single do_row body
// and reuses it across OUR * UR iterations of the surrounding
// #pragma-unrolled loops. This is the main lever for keeping
// kernel_<profile>_v<N>.cu compile times manageable when OUR is large.
//
// info.cov, inormc, dfc, dgc are arrays held in registers; inormr, dfr,
// dgr are scalars (the caller indexes the row arrays).
/////////////////////////////////////////////////////
template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          typename DISTANCE_TYPE, typename DerivedInputType, int DiagsPerThread,
          typename DerivedSmem, typename DistColArray, typename InputColArray,
          typename IndexColArray>
__device__ inline FORCE_INLINE void do_row(
    int outer_row_iter, int row_iter, const SCAMPKernelInputArgs<double> &args,
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
    dist[i] =
        static_cast<DISTANCE_TYPE>(info.cov[i] * inormc[row_iter + i] * inormr);
    info.cov[i] =
        info.cov[i] + dfc[row_iter + i] * dgr + dgc[row_iter + i] * dfr;
  }
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    // Per-cell reduction (mirrors CPU update_mp): every (row, col) distance is
    // bucketed and max-accumulated into the block's smem cell grid. No
    // per-column/row profile is kept for matrix summary.
    const int gr = info.global_row + outer_row_iter;
#pragma unroll unrolled_diags
    for (int i = 0; i < unrolled_diags; ++i) {
      ms_accumulate_cell(
          smem, static_cast<double>(gr),
          static_cast<double>(info.global_col + outer_row_iter + i),
          static_cast<float>(dist[i]), args);
    }
  } else {
    if constexpr (COMPUTE_COLS) {
      merge_to_column<PROFILE_TYPE, DiagsPerThread>(outer_row_iter, args, info,
                                                    smem, distc, dist, idxc);
    }
    if constexpr (COMPUTE_ROWS) {
      merge_to_row<PROFILE_TYPE, DISTANCE_TYPE, DerivedInputType,
                   DiagsPerThread>(outer_row_iter, args, info, smem, dist, distr,
                                   idxr);
    }
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
//
// The outer (OUR/UR-iter) and inner (UR-iter) loops are #pragma-unrolled
// regular for-loops (not constexpr-recursive for_<N>(...) lambdas) so the
// surrounding do_row / update_cols / update_rows / merge_to_* templates
// only get one instantiation per (variant, precision, row/col mode)
// regardless of OUR. Keeps per-variant compile time roughly independent
// of OUR; the recursive-lambda form would pay an OUR-proportional cost
// (template instantiation per outer-loop k) for no codegen win.
///////////////////////////////////////////////////////////////////////////////
template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          typename DISTANCE_TYPE, int DiagsPerThread, int UnrolledRows,
          int OuterUnrolledRows, typename DerivedDataType, typename DerivedSmem>
__device__ void do_iteration_fast(
    const SCAMPKernelInputArgs<double> &args,
    SCAMPThreadInfo<DerivedDataType, DiagsPerThread> &info, DerivedSmem &smem) {
  // The "natural" inner_unrolled_cols would be DPT + UR - 1: the smallest
  // window that holds DPT cols per row across UR rows. We pad it to DPT + UR
  // so the sliding-window refill lands at byte offset
  //   (threadIdx.x*DPT + j*UR + DPT) * sizeof(T)
  // = DPT*(threadIdx.x + 1) * sizeof(T) + j * UR * sizeof(T)
  // which is UR*sizeof(T)-aligned for every (DPT, UR) tuple where DPT % UR
  // == 0 (true for all current variants). That lets vec_load emit
  // ld.shared.v2.f32 / v4.f32 / v2.f64 for the slide instead of scalar
  // ld.shared.{f32,f64}. The extra slot at the right edge is never read by
  // do_row (whose max access is row_iter + DPT - 1 <= inner - 2), so the
  // only cost is one wasted register per (dfc, dgc, inormc) = 3 per thread.
  static_assert(DiagsPerThread % UnrolledRows == 0,
                "DPT must be divisible by UR for aligned slide-loads.");
  constexpr int inner_unrolled_cols = DiagsPerThread + UnrolledRows;
  constexpr int unrolled_cols = DiagsPerThread + OuterUnrolledRows - 1;
  constexpr int outer_iters = OuterUnrolledRows / UnrolledRows;
  static_assert(outer_iters * UnrolledRows == OuterUnrolledRows,
                "OuterUnrolledRows must be divisible by UnrolledRows.");

  using SmemDataType = typename DerivedSmem::DataType;
  Eigen::Array<SmemDataType, inner_unrolled_cols, 1> dfc, dgc, inormc;
  DISTANCE_TYPE init = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  Eigen::Array<DISTANCE_TYPE, unrolled_cols, 1> distc =
      Eigen::Array<DISTANCE_TYPE, unrolled_cols, 1>::Constant(init);
  Eigen::Array<DISTANCE_TYPE, OuterUnrolledRows, 1> distr =
      Eigen::Array<DISTANCE_TYPE, OuterUnrolledRows, 1>::Constant(init);
  Eigen::Array<unsigned int, unrolled_cols, 1> idxc;
  Eigen::Array<unsigned int, OuterUnrolledRows, 1> idxr;

  // Initial load of the column sliding-window from smem. smem.df_col.data()
  // + info.local_col is aligned to DiagsPerThread * sizeof(SmemDataType)
  // bytes (info.local_col = threadIdx.x * DiagsPerThread). vec_load emits
  // ld.shared.v4.f32 / v2.f32 / v2.f64 etc instead of N scalar
  // ld.shared.{f32,f64} that Eigen::Map::segment<> assignments compile to.
  constexpr int kColAlignBytes = DiagsPerThread * sizeof(SmemDataType);
  vec_load<inner_unrolled_cols, kColAlignBytes, SmemDataType>(
      smem.df_col.data() + info.local_col, dfc.data());
  vec_load<inner_unrolled_cols, kColAlignBytes, SmemDataType>(
      smem.dg_col.data() + info.local_col, dgc.data());
  vec_load<inner_unrolled_cols, kColAlignBytes, SmemDataType>(
      smem.inorm_col.data() + info.local_col, inormc.data());

  // Outer loop: process OuterUnrolledRows rows in batches of UnrolledRows.
  // #pragma-unrolled regular for so do_row / update_cols / update_rows get
  // a single template instantiation regardless of OUR (see file header).
#pragma unroll
  for (int j = 0; j < outer_iters; ++j) {
    if (j > 0) {
      // Slide the column window left by UnrolledRows and load the next
      // UnrolledRows columns into the right edge. The if (j > 0) is
      // runtime but constant-folded away in iter 0 once nvcc unrolls.
      // Refill smem offset = local_col + j*UR + (inner - UR) = ... + DPT,
      // which is UR-aligned (DPT % UR == 0 enforced above), so vec_load
      // emits ld.shared.v{UR}.{f32,f64} instead of scalar loads.
      dfc.template segment<inner_unrolled_cols - UnrolledRows>(0) =
          dfc.template segment<inner_unrolled_cols - UnrolledRows>(
              UnrolledRows);
      dgc.template segment<inner_unrolled_cols - UnrolledRows>(0) =
          dgc.template segment<inner_unrolled_cols - UnrolledRows>(
              UnrolledRows);
      inormc.template segment<inner_unrolled_cols - UnrolledRows>(0) =
          inormc.template segment<inner_unrolled_cols - UnrolledRows>(
              UnrolledRows);
      constexpr int kSlideAlignBytes = UnrolledRows * sizeof(SmemDataType);
      vec_load<UnrolledRows, kSlideAlignBytes, SmemDataType>(
          smem.df_col.data() + info.local_col + j * UnrolledRows +
              (inner_unrolled_cols - UnrolledRows),
          dfc.data() + (inner_unrolled_cols - UnrolledRows));
      vec_load<UnrolledRows, kSlideAlignBytes, SmemDataType>(
          smem.dg_col.data() + info.local_col + j * UnrolledRows +
              (inner_unrolled_cols - UnrolledRows),
          dgc.data() + (inner_unrolled_cols - UnrolledRows));
      vec_load<UnrolledRows, kSlideAlignBytes, SmemDataType>(
          smem.inorm_col.data() + info.local_col + j * UnrolledRows +
              (inner_unrolled_cols - UnrolledRows),
          inormc.data() + (inner_unrolled_cols - UnrolledRows));
    }
    // smem.df_row.data() + info.local_row + j*UnrolledRows is aligned to
    // UnrolledRows * sizeof(SmemDataType) bytes: info.local_row is a
    // multiple of OuterUnrolledRows (incremented by OUR at end of
    // do_iteration_fast); OUR is a multiple of UR; j*UR is a multiple of
    // UR. So total is a multiple of UR.
    constexpr int kRowAlignBytes = UnrolledRows * sizeof(SmemDataType);
    Eigen::Array<SmemDataType, UnrolledRows, 1> dfr, dgr, inormr;
    vec_load<UnrolledRows, kRowAlignBytes, SmemDataType>(
        smem.df_row.data() + info.local_row + j * UnrolledRows, dfr.data());
    vec_load<UnrolledRows, kRowAlignBytes, SmemDataType>(
        smem.dg_row.data() + info.local_row + j * UnrolledRows, dgr.data());
    vec_load<UnrolledRows, kRowAlignBytes, SmemDataType>(
        smem.inorm_row.data() + info.local_row + j * UnrolledRows,
        inormr.data());
#pragma unroll
    for (int k = 0; k < UnrolledRows; ++k) {
      do_row<PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, DISTANCE_TYPE,
             DerivedDataType, DiagsPerThread>(
          /*outer_row_iter=*/j * UnrolledRows + k, /*row_iter=*/k, args, info,
          smem, distc, distr[j * UnrolledRows + k], inormc, dfc, dgc, inormr[k],
          dfr[k], dgr[k], idxc, idxr[j * UnrolledRows + k]);
    }
    // MATRIX_SUMMARY accumulates directly into the smem cell grid inside
    // do_row, so it keeps no per-column/row profile to flush here.
    if constexpr (COMPUTE_COLS && PROFILE_TYPE != PROFILE_TYPE_MATRIX_SUMMARY) {
      update_cols<UnrolledRows, PROFILE_TYPE, DerivedDataType, DiagsPerThread>(
          /*start_index=*/j * UnrolledRows, args, info, smem, distc, idxc);
    }
    if constexpr (COMPUTE_ROWS && PROFILE_TYPE != PROFILE_TYPE_MATRIX_SUMMARY) {
      update_rows<UnrolledRows, PROFILE_TYPE, DISTANCE_TYPE, DerivedDataType,
                  DiagsPerThread>(/*row_iter=*/j * UnrolledRows, args, info,
                                  smem, distr, idxr);
    }
  }

  // Flush the trailing tail of distc (positions OuterUnrolledRows
  // through unrolled_cols-1 hold bests merged but never atomic-flushed
  // by the per-batch update above). MATRIX_SUMMARY keeps no distc tail.
  if constexpr (COMPUTE_COLS && PROFILE_TYPE != PROFILE_TYPE_MATRIX_SUMMARY) {
    update_cols<unrolled_cols - OuterUnrolledRows, PROFILE_TYPE,
                DerivedDataType, DiagsPerThread>(
        /*start_index=*/OuterUnrolledRows, args, info, smem, distc, idxc);
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
        reinterpret_cast<uint64_t *>(smem.local_mp_row.data()) + info.local_row,
        dist_row, idx_row, -2);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    do_atomicAdd<double, ATOMIC_BLOCK>(
        smem.local_mp_row.data() + info.local_row, dist_row);
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "reduce_row not implemented for profile type.");
  }
}

// iter is runtime (diagonal index inside the edge row); the surrounding
// caller #pragma-unrolls so each iter value lands as a constant in the
// emitted code.
template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          typename DerivedSmemType, typename DerivedDataType,
          int DiagsPerThread, typename DerivedDist, typename DerivedDist4>
__device__ inline void reduce_edge(
    int iter, const SCAMPKernelInputArgs<double> &args,
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
          fAtomicMax<ATOMIC_BLOCK>(
              smem.local_mp_col.data() + info.local_col + iter, dist[iter]);
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
  Eigen::Array<DISTANCE_TYPE, DiagsPerThread, 1> dist;
#pragma unroll DiagsPerThread
  for (int i = 0; i < DiagsPerThread; ++i) {
    dist[i] = static_cast<DISTANCE_TYPE>(
        info.cov[i] * smem.inorm_col[info.local_col + i] * inormr);
    info.cov[i] = info.cov[i] + smem.df_col[info.local_col + i] * dgr +
                  smem.dg_col[info.local_col + i] * dfr;
  }

  if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    // Per-cell reduction with the same bounds check reduce_edge applies.
#pragma unroll DiagsPerThread
    for (int i = 0; i < DiagsPerThread; ++i) {
      if (info.global_col + i < static_cast<uint32_t>(args.n_x) &&
          diag + i < num_diags) {
        ms_accumulate_cell(smem, static_cast<double>(info.global_row),
                           static_cast<double>(info.global_col + i),
                           static_cast<float>(dist[i]), args);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < DiagsPerThread; ++i) {
      reduce_edge<PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, DerivedSmemType,
                  DerivedDataType, DiagsPerThread>(/*iter=*/i, args, info, smem,
                                                   dist, dist_row, idx_row, diag,
                                                   num_diags);
    }

    if constexpr (COMPUTE_ROWS) {
      reduce_row<PROFILE_TYPE, DerivedDataType, DiagsPerThread>(
          args, info, smem, dist_row, idx_row);
    }
  }
}
