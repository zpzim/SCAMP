#pragma once

// Constexpr for loop definition. Unrolled by definition.
template <std::size_t N>
struct num {
  static const constexpr auto value = N;
};

template <class F, std::size_t... Is>
__device__ void for_(F func, std::index_sequence<Is...>) {
  using expander = int[];
  (void)expander{0, ((void)func(num<Is>{}), 0)...};
}

template <std::size_t N, typename F>
__device__ void for_(F func) {
  for_(func, std::make_index_sequence<N>());
}

// Helper function to compute arraywise maximums.
// Ignores nan, returns -2 if the entire array contains NaN.
template <typename Derived, typename ScalarType = typename Derived::Scalar>
__device__ inline ScalarType max_dist(const Eigen::ArrayBase<Derived>& dist) {
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

// Helper function to compute arraywise maximums and also sets idx to the index
// of the maximum element. Ignores nan, returns -2 if the entire array contains
// NaN.
template <typename Derived, typename ScalarType = typename Derived::Scalar>
__device__ inline ScalarType max_dist(const Eigen::ArrayBase<Derived>& dist,
                                      int& idx) {
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
// UPDATE_ROW:
// C: 0 1 2 3 4 5 6
// R0 X X X X
// R1   X X X X
// R2     X X X X
// R3       X X X X
// One definition required for each profile type
// For each "thread row" of 4 distances computed, this function performs a
// reduction on those distances and updates the mp value corresponding with that
// row For example, in the diagram above, for iter = 1, it is assumed that the
// distances held in 'dist' correspond to the distances for R1 (corresponding to
// columns '1,2,3, and 4') This function takes those distances and merges them
// into a single "best" value for the row. This computation is dependant on the
// type of profile being computed For 1NN MP the result is finding the maximum
// of 'dist' (and the index, if required) For SUM MP the result is finding the
// sum of the values in 'dist' greater than the threshold
// ...
// TO ADD A NEW PROFILE TYPE ADD A BRANCH IN THIS FUNCTION FOR THAT TYPE
// Perform any profile specific computations on 'dist'
// Perform reduction of 'dist'
// Update shared memory using result of reduction
//////////////////////////////////////////////////////

template <int iter, SCAMPProfileType PROFILE_TYPE, typename DISTANCE_TYPE,
          typename InputDataType, typename DerivedSmem, typename DistRowArray>
__device__ inline void update_row(const SCAMPKernelInputArgs<double>& args,
                                  const SCAMPThreadInfo<InputDataType>& info,
                                  DerivedSmem& smem,
                                  const Eigen::ArrayBase<DistRowArray>& dist,
                                  const float curr_mp_row_val) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    DISTANCE_TYPE d = max_dist(dist);
    fAtomicMax_check<ATOMIC_BLOCK>(
        smem.local_mp_row.data() + info.local_row + iter, d, curr_mp_row_val);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    int idx;
    DISTANCE_TYPE d = max_dist(dist, idx);
    idx += info.global_col + iter;
    // Coalesce all row updates to lane 0 of each warp and atomically update
    unsigned mask = __ballot_sync(0xffffffff, d > curr_mp_row_val);
    int count = __popc(mask);
    if (count > 0) {
      mp_entry local_max;
      local_max.floats[0] = d;
      local_max.ints[1] = idx;
#pragma unroll
      for (int i = 16; i >= 1; i /= 2) {
        mp_entry other;
        other.ulong = __shfl_down_sync(0xffffffff, local_max.ulong, i);
        if (other.floats[0] > local_max.floats[0]) {
          local_max.ulong = other.ulong;
        }
      }
      if ((threadIdx.x & 0x1f) == 0) {
        MPatomicMax<ATOMIC_BLOCK>(
            smem.local_mp_row.data() + info.local_row + iter,
            local_max.floats[0], local_max.ints[1]);
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    // Coalesce all row updates to lane 0 of each warp and atomically update
    // This way is more efficient than atomics when we expect a lot of updates
    DISTANCE_TYPE sum = (dist > args.opt.threshold).select(dist, 0).sum();
#pragma unroll
    for (int i = 16; i >= 1; i /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    if ((threadIdx.x & 0x1f) == 0) {
      do_atomicAdd<double, ATOMIC_BLOCK>(
          smem.local_mp_row.data() + info.local_row + iter,
          static_cast<double>(sum));
    }
  } else {
    static_assert(PROFILE_TYPE != -1,
                  "update_row not implemented for profile type.");
  }
}

//////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// MERGE_TO_COLUMN:
// C: 0 1 2 3 4 5 6
// R0 X X X X
// R1   X X X X
// R2     X X X X
// R3       X X X X
// One definition required for each profile type
// For each "thread row" of 4 distances computed, this function merges those
// distances with the appropriate best-so-far values. For example, in the
// diagram above, for iter = 1, it is assumed that the distances held in
// 'dists_to_merge' correspond to the distances for R1 (corresponding to columns
// '1,2,3, and 4') This function takes those distances and merges them into
// 'best_so_far' column values for the tile. This computation is dependant on
// the type of profile being computed: For 1NN MP the result is finding the
// pairwise maximum of 'dists_to_merge', and 'best_so_far' according to the
// specific row 'iter' (and the index, if required) For SUM MP the result is
// finding adding each distance (greater than the threshold) in
// 'dists_to_merge', with the corresponding 'best_so_far' values, and storing
// the result in 'best_so_far'
// ...
// TO ADD A NEW PROFILE TYPE ADD A BRANCH IN THIS FUNCTION FOR THAT TYPE
// Perform any profile specific computations on the dists_to_merge
// Perform pairwise merge of 'dists_to_merge' and 'best_so_far[iter, iter+3]'
// storing the result in 'best_so_far[iter, iter+3]'
/////////////////////////////////////////////////////////////////////////

template <int iter, SCAMPProfileType PROFILE_TYPE, typename DerivedDataType,
          typename DerivedSmem, typename ColDistArray, typename RowDistArray,
          typename ColIndexArray>
__device__ inline void merge_to_column(
    const SCAMPKernelInputArgs<double>& args,
    const SCAMPThreadInfo<DerivedDataType>& info, DerivedSmem smem,
    Eigen::ArrayBase<ColDistArray>& best_so_far,
    const Eigen::ArrayBase<RowDistArray>& dists_to_merge,
    Eigen::ArrayBase<ColIndexArray>& best_so_far_index) {
  static_assert(RowDistArray::RowsAtCompileTime == DIAGS_PER_THREAD);
  static_assert(ColDistArray::RowsAtCompileTime ==
                ColIndexArray::RowsAtCompileTime);
  static_assert(ColDistArray::RowsAtCompileTime == unrolled_cols);
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
#pragma unroll DIAGS_PER_THREAD
    for (int i = 0; i < DIAGS_PER_THREAD; ++i) {
      if (dists_to_merge[i] > best_so_far[iter + i]) {
        best_so_far[iter + i] = dists_to_merge[i];
      }
    }

  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
#pragma unroll DIAGS_PER_THREAD
    for (int i = 0; i < DIAGS_PER_THREAD; ++i) {
      if (dists_to_merge[i] > best_so_far[iter + i]) {
        best_so_far[iter + i] = dists_to_merge[i];
        best_so_far_index[iter + i] = info.global_row + iter;
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    best_so_far.segment<DIAGS_PER_THREAD>(iter) +=
        (dists_to_merge > args.opt.threshold).select(dists_to_merge, 0);
  } else {
    static_assert(PROFILE_TYPE != -1,
                  "merge_to_column not implemented for profile type.");
  }
}

///////////////////////////////////////////////////////////////////
// UPDATE_COLS:
// C: 0 1 2 3 4 5 6
// R0 X X X X
// R1   X X X X
// R2     X X X X
// R3       X X X X
// This function takes the thread-local best so far values for each column
// (0,1,2,3,4,5,and 6) and merges them with the shared-memory MP for each
//////////////////////////////////////////////////////////////////

template <SCAMPProfileType PROFILE_TYPE, typename DerivedInputDataType,
          typename DerivedSmemType, typename ColDistArray,
          typename ColIndexArray>
__device__ inline void update_cols(const SCAMPKernelInputArgs<double>& args,
                                   SCAMPThreadInfo<DerivedInputDataType>& info,
                                   DerivedSmemType& smem,
                                   Eigen::ArrayBase<ColDistArray>& distc,
                                   Eigen::ArrayBase<ColIndexArray>& idxc) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    Eigen::Array<float, unrolled_cols, 1> mp_col_check =
        smem.local_mp_col.segment<unrolled_cols>(info.local_col);
// Check the best-so-far column and update distance if necessary
#pragma unroll unrolled_cols
    for (int i = 0; i < unrolled_cols; ++i) {
      float old = fAtomicMax_check<ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + i, distc[i],
          mp_col_check[i]);
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    Eigen::Array<float, unrolled_cols, 1> mp_col_check;
    {
      Eigen::Array<uint64_t, unrolled_cols, 1> temp =
          smem.local_mp_col.segment<unrolled_cols>(info.local_col);
#pragma unroll unrolled_cols
      for (int i = 0; i < unrolled_cols; ++i) {
        mp_entry e;
        e.ulong = temp[i];
        mp_col_check[i] = e.floats[0];
      }
    }

// Check the best-so-far column and update distance/index if necessary
#pragma unroll unrolled_cols
    for (int i = 0; i < unrolled_cols; ++i) {
      MPatomicMax_check<ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + i, distc[i], idxc[i],
          mp_col_check[i]);
    }

  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
// Add the current sum that this thread has computed to the shared sum across
// the entire thread block
#pragma unroll unrolled_cols
    for (int i = 0; i < unrolled_cols; ++i) {
      do_atomicAdd<double, ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + i, distc[i]);
    }

  } else {
    static_assert(PROFILE_TYPE != -1,
                  "update_cols not implemented for profile type.");
  }
}

/////////////////////////////////////////////////////
// DO_ROW:
// C: 0 1 2 3 4 5 6
// R0 X X X X
// R1   X X X X
// R2     X X X X
// R3       X X X X
// Computes a single row of the the distances for the tile above, and performs a
// row-wise and partial column wise reduction on the distances For example if
// iter == 2, this will compute the distances corresponding with R2 and columns
// 2,3,4,and 5, merge the distances into a single value for the MP value
// associated with R2, and perform a pairwise reduction for the best so far
// values associated with columns 2,3,4, and 5.
// DO NOT EDIT this function unless you are sure you know what you are doing as
// it is templated and used by ALL profile computations.
//////////////////////////////////////////////////////////
template <int iter, SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, typename DISTANCE_TYPE, typename DerivedInputType,
          typename DerivedSmem, typename DistColArray, typename InputColArray,
          typename InputRowArray, typename IndexColArray,
          typename RowType = typename InputRowArray::Scalar,
          typename ColType = typename InputColArray::Scalar>
__device__ inline FORCE_INLINE void do_row(
    const SCAMPKernelInputArgs<double>& args,
    SCAMPThreadInfo<DerivedInputType>& info, DerivedSmem& smem,
    Eigen::ArrayBase<DistColArray>& distc,
    const Eigen::ArrayBase<InputColArray>& inormc,
    const Eigen::ArrayBase<InputColArray>& dfc,
    const Eigen::ArrayBase<InputColArray>& dgc,
    const Eigen::ArrayBase<InputRowArray>& inormr,
    const Eigen::ArrayBase<InputRowArray>& dfr,
    const Eigen::ArrayBase<InputRowArray>& dgr, const float curr_mp_row_val,
    Eigen::ArrayBase<IndexColArray>& idxc) {
  static_assert(std::is_same<RowType, ColType>::value);

  // Compute the correlation values for the current tile row
  Eigen::Array<DISTANCE_TYPE, DIAGS_PER_THREAD, 1> dist;
#pragma unroll DIAGS_PER_THREAD
  for (int i = 0; i < DIAGS_PER_THREAD; ++i) {
    dist[i] = info.cov[i] * inormc[iter + i] * inormr[iter];
    info.cov[i] =
        info.cov[i] + dfc[iter + i] * dgr[iter] + dgc[iter + i] * dfr[iter];
  }

  // Update the column best-so-far values
  if constexpr (COMPUTE_COLS) {
    merge_to_column<iter, PROFILE_TYPE>(args, info, smem, distc, dist, idxc);
  }

  // Perform any updates for this tile row and commit to the shared-memory
  // matrix profile
  if constexpr (COMPUTE_ROWS) {
    update_row<iter, PROFILE_TYPE, DISTANCE_TYPE>(args, info, smem, dist,
                                                  curr_mp_row_val);
  }
}

///////////////////////////////////////////////////////////////////////////////
// OPTIMIZED CODE PATH:
// do_iteration_fast is the optimized matrix profile code path which computes
// one row of work for a single thread. It is specialized for each profile type
// that is computed.
// This function computes a 4x4 block of the distance matrix by calling
// do_row() four separate times.
// We are computing a tile that looks like this:
// C: 0 1 2 3 4 5 6
// R0 X X X X
// R1   X X X X
// R2     X X X X
// R3       X X X X
// For 4 diagonals unrolled 4 times we compute a total of 16 distances.
// These distances cover 4 possible rows and 7 possible columns.
// Each row of 4 distances is computed via the do_row<>() function
///////////////////////////////////////////////////////////////////////////////
// Processes 4 iterations of the inner loop. Each thread computes 4 distances
// per iteration (x,y), (x+1,y), (x+2,y), and (x+3,y) This function assumes that
// the edge cases that occur on the edge of the distance matrix are not present.
// This is the faster path, with less conditional branching.
// DO NOT EDIT this function unless you are sure you know what you are doing, as
// it is called for every kernel with various template parameters. It is also
// written to be highly performant, as this code is the main bottleneck in the
// computation. If you have an optimization for this segment of code, it will
// make ALL profiles faster.
template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          typename DISTANCE_TYPE, typename DerivedDataType,
          typename DerivedSmem>
void __device__ do_iteration_fast(const SCAMPKernelInputArgs<double>& args,
                                  SCAMPThreadInfo<DerivedDataType>& info,
                                  DerivedSmem& smem) {
  Eigen::Array<DerivedDataType, unrolled_cols, 1> dfc, dgc, inormc;
  Eigen::Array<DerivedDataType, unrolled_rows, 1> dgr, dfr, inormr;
  Eigen::Array<float, unrolled_rows, 1> mp_row_check;
  DISTANCE_TYPE init = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  Eigen::Array<DISTANCE_TYPE, unrolled_cols, 1> distc =
      Eigen::Array<DISTANCE_TYPE, unrolled_cols, 1>::Constant(init);
  Eigen::Array<unsigned int, unrolled_cols, 1> idxc;

  dfc = smem.df_col.segment<unrolled_cols>(info.local_col);
  dgc = smem.dg_col.segment<unrolled_cols>(info.local_col);
  inormc = smem.inorm_col.segment<unrolled_cols>(info.local_col);

  dfr = smem.df_row.segment<unrolled_rows>(info.local_row);
  dgr = smem.dg_row.segment<unrolled_rows>(info.local_row);
  inormr = smem.inorm_row.segment<unrolled_rows>(info.local_row);

  // For NN profiles we need to do a vectorized load to pull the best-so-far
  // values from cache
  if constexpr (COMPUTE_ROWS) {
    if constexpr (PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS ||
                  PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                  PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX) {
      Eigen::Array<uint64_t, unrolled_rows, 1> temp =
          smem.local_mp_row.segment<unrolled_rows>(info.local_row);
#pragma unroll unrolled_rows
      for (int i = 0; i < unrolled_rows; ++i) {
        mp_entry e;
        e.ulong = temp[i];
        mp_row_check[i] = e.floats[0];
      }
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
      mp_row_check = smem.local_mp_row.segment<unrolled_rows>(info.local_row);
    }
  }

  // Generate and coalesce distances into profile
  for_<unrolled_rows>([&](auto i) {
    do_row<i.value, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, DISTANCE_TYPE>(
        args, info, smem, distc, inormc, dfc, dgc, inormr, dfr, dgr,
        mp_row_check[i.value], idxc);
  });

  // Update the column wise matrix profile with the best-so-far
  if constexpr (COMPUTE_COLS) {
    update_cols<PROFILE_TYPE>(args, info, smem, distc, idxc);
  }

  // Advance counters
  info.local_col += unrolled_rows;
  info.local_row += unrolled_rows;
  info.global_col += unrolled_rows;
  info.global_row += unrolled_rows;
}

/////////////////////////////////////////////////////////////////////////
//  EDGE COMPUTATION
//////////////////////////////////////////////////////////////////////

template <SCAMPProfileType PROFILE_TYPE, typename DerivedDataType,
          typename DerivedDist, typename DerivedSmemType>
__device__ inline void reduce_row(const SCAMPKernelInputArgs<double>& args,
                                  const SCAMPThreadInfo<DerivedDataType>& info,
                                  DerivedSmemType& smem, DerivedDist dist_row,
                                  uint32_t idx_row) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_row.data() + info.local_row,
                                   dist_row, -2);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    MPatomicMax_check<ATOMIC_BLOCK>(smem.local_mp_row.data() + info.local_row,
                                    dist_row, idx_row, -2);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    do_atomicAdd<double, ATOMIC_BLOCK>(
        smem.local_mp_row.data() + info.local_row, dist_row);
  } else {
    static_assert(PROFILE_TYPE != -1,
                  "reduce_row not implemented for profile type.");
  }
}

template <int iter, SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, typename DerivedSmemType, typename DerivedDataType,
          typename DerivedDist, typename DerivedDist4>
__device__ inline void reduce_edge(const SCAMPKernelInputArgs<double>& args,
                                   const SCAMPThreadInfo<DerivedDataType>& info,
                                   DerivedSmemType& smem,
                                   const Eigen::ArrayBase<DerivedDist4>& dist,
                                   DerivedDist& dist_row, uint32_t& idx_row,
                                   int diag, int num_diags) {
  // Check if it is safe to do this iteration (we may have gone over the edge)
  if (info.global_col + iter < args.n_x && diag + iter < num_diags) {
    if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
      if constexpr (COMPUTE_ROWS) {
        dist_row = fmaxf(dist_row, dist[iter]);
      }
      if constexpr (COMPUTE_COLS) {
        fAtomicMax_check<ATOMIC_BLOCK>(
            smem.local_mp_col.data() + info.local_col + iter, dist[iter], -2);
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
        MPatomicMax_check<ATOMIC_BLOCK>(
            smem.local_mp_col.data() + info.local_col + iter, dist[iter],
            info.global_row, -2);
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
      static_assert(PROFILE_TYPE != -1,
                    "reduce_edge not implemented for profile type.");
    }
  }
}

template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          typename DISTANCE_TYPE, typename DerivedSmemType,
          typename DerivedDataType>
__device__ inline void do_row_edge(const SCAMPKernelInputArgs<double>& args,
                                   SCAMPThreadInfo<DerivedDataType>& info,
                                   DerivedSmemType& smem, int diag,
                                   int num_diags) {
  DISTANCE_TYPE dist_row = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  Eigen::Array<DISTANCE_TYPE, DIAGS_PER_THREAD, 1> distc =
      Eigen::Array<DISTANCE_TYPE, DIAGS_PER_THREAD, 1>::Constant(dist_row);
  uint32_t idx_row = 0;
  DerivedDataType inormr = smem.inorm_row[info.local_row];
  DerivedDataType dgr = smem.dg_row[info.local_row];
  DerivedDataType dfr = smem.df_row[info.local_row];

  // Compute the next set of distances. Only a single row. Note this may compute
  // garbage for values beyond the edge of the array.
  Eigen::Array<DISTANCE_TYPE, DIAGS_PER_THREAD, 1> dist =
      (info.cov * smem.inorm_col.segment<DIAGS_PER_THREAD>(info.local_col) *
       inormr)
          .template cast<DISTANCE_TYPE>();

  // Update cov and compute the next distance values. Note this may compute
  // garbage for values beyond the edge of the array.
  info.cov += smem.df_col.segment<DIAGS_PER_THREAD>(info.local_col) * dgr +
              smem.dg_col.segment<DIAGS_PER_THREAD>(info.local_col) * dfr;

  for_<DIAGS_PER_THREAD>([&](auto i) {
    reduce_edge<i.value, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS>(
        args, info, smem, dist, dist_row, idx_row, diag, num_diags);
  });

  if constexpr (COMPUTE_ROWS) {
    reduce_row<PROFILE_TYPE>(args, info, smem, dist_row, idx_row);
  }
}
