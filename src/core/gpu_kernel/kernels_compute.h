#pragma once
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
//////////////////////////////////////////////////////


template <int iter, SCAMPProfileType PROFILE_TYPE, typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE, typename DATA_TYPE, typename SMEM_TYPE, typename DerivedDist>
__device__ inline void update_row(
    const SCAMPThreadInfo<DATA_TYPE> info,
    SMEM_TYPE smem,
    const Eigen::ArrayBase<DerivedDist>& dist, const float curr_mp_row_val,
    const OptionalArgs args) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    DISTANCE_TYPE d = dist.maxCoeff();
    fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_row.data() + info.local_row + iter, d,
                                 curr_mp_row_val);
  } else if constexpr(PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX || PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY || PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    unsigned int idx;
    DISTANCE_TYPE d = dist.maxCoeff(&idx);
    idx += info.global_col + iter;
    MPatomicMax_check<ATOMIC_BLOCK>(
        (uint64_t *)(smem.local_mp_row.data() + info.local_row + iter), d, idx,
        curr_mp_row_val);
  } else if constexpr(PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    // Coalesce all row updates to lane 0 of each warp and atomically update
    // This way is more efficient than atomics when we expect a lot of updates
    DISTANCE_TYPE sum = 0;
    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
      if (dist[i] > args.threshold) {
        sum += dist[i];
      }
    }
    #pragma unroll
    for (int i = 16; i >= 1; i /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    if ((threadIdx.x & 0x1f) == 0) {
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
          smem.local_mp_row.data() + info.local_row + iter,
          static_cast<PROFILE_DATA_TYPE>(sum));
    }
  } else {
    // Missing implementation.   
  }
}

/*
// TO ADD A NEW PROFILE TYPE IMPLEMENT THIS FUNCTION FOR THAT TYPE

// UPDATE_ROW when PROFILE_TYPE == PROFILE_TYPE_???
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename DATA_TYPE, typename DISTANCE_TYPE>
__device__ inline void update_row(
    const SCAMPThreadInfo<DATA_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_???> smem,
    const DISTANCE_TYPE dist[4], const float curr_mp_row_val,
    const OptionalArgs args) {
  // YOUR CODE HERE
  // Perform any profile specific computations on 'dist'
  // Perform reduction of 'dist'
  // Update shared memory using result of reduction
}
*/

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
/////////////////////////////////////////////////////////////////////////

template <int iter, SCAMPProfileType PROFILE_TYPE, typename DATA_TYPE, typename SMEM_TYPE, typename Derived7Dist, typename Derived4Dist, typename Derived7uint>
__device__ inline void merge_to_column(
    const SCAMPThreadInfo<DATA_TYPE>& info,
    SMEM_TYPE smem,
    Eigen::ArrayBase<Derived7Dist>& best_so_far, const Eigen::ArrayBase<Derived4Dist>& dists_to_merge,
    Eigen::ArrayBase<Derived7uint>& best_so_far_index, const OptionalArgs& args) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    best_so_far.segment<4>(iter) = dists_to_merge.cwiseMax(best_so_far.segment<4>(iter));
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX || PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY || PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
      if (dists_to_merge[i] > best_so_far[iter + i]) {
        best_so_far[iter + i] = dists_to_merge[i];
        best_so_far_index[iter + i] = info.global_row + iter;
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
      if (dists_to_merge[i] > args.threshold) {
        best_so_far[iter + i] += dists_to_merge[i];
      }
    }
  } else {
    // Unimplemented
    //static_assert(false, "merge_to_column not implemented for profile type: PROFILE_TYPE")
  }
}

/*
// TO ADD A NEW PROFILE TYPE IMPLEMENT THIS FUNCTION FOR THAT TYPE

// MERGE_TO_COLUMN when PROFILE_TYPE == PROFILE_TYPE_???
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename DATA_TYPE, typename DISTANCE_TYPE>
__device__ inline void merge_to_column(
    const SCAMPThreadInfo<DATA_TYPE> info,
    const SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_???> smem,
    DISTANCE_TYPE best_so_far[7], const DISTANCE_TYPE dists_to_merge[4],
    unsigned int best_so_far_index[7], const OptionalArgs args) {
  // YOUR CODE HERE
  // Perform any profile specific computations on the dists_to_merge
  // Perform pairwise merge of 'dists_to_merge[i]' and 'best_so_far[iter + i]'
storing the result in 'best_so_far[iter+i]'
}
*/

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

template <SCAMPProfileType PROFILE_TYPE, typename PROFILE_DATA_TYPE, typename DerivedInputDataType, typename DerivedSmemType, typename Derived7Dist, typename Derived7Uint>
__device__ inline void update_cols(
    SCAMPThreadInfo<DerivedInputDataType> info,
    DerivedSmemType smem,
    Eigen::ArrayBase<Derived7Dist>& distc, Eigen::ArrayBase<Derived7Uint>& idxc) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    Eigen::Array<float, 7, 1> mp_col_check = smem.local_mp_col.segment<7>(info.local_col);

    // Check the best-so-far column and update distance if necessary
    #pragma unroll 7
    for (int i = 0; i < 7; ++i) {
      fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col.data() + info.local_col + i,
                                     distc[i], mp_col_check[i]);
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX || PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY || PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    Eigen::Array<float, 7, 1> mp_col_check;
    {
      Eigen::Array<uint64_t, 7, 1> temp = smem.local_mp_col.segment<7>(info.local_col);
      mp_entry e;
      #pragma unroll 7
      for (int i = 0; i < 7; ++i) {
        e.ulong = temp[i];
        mp_col_check[i] = e.floats[0];
      }
    }
   
    // Check the best-so-far column and update distance/index if necessary
    #pragma unroll 7
    for (int i = 0; i < 7; ++i) {
      MPatomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col.data() + info.local_col + i,
                                      distc[i], idxc[i], mp_col_check[i]);
    }

  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    // Add the current sum that this thread has computed to the shared sum across
    // the entire thread block
    #pragma unroll 7
    for (int i = 0; i < 7; ++i) {
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + i, distc[i]);
    }

  } else {
    //static_assert(false, "update_cols not implemented for profile type: PROFILE_TYPE")
    // Unimplemented.
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
// values associated with columns 2,3,4, and 5 DO NOT EDIT this function unless
// you are sure you know what you are doing as it is templated and used by ALL
// profile computations.
//////////////////////////////////////////////////////////
template <int iter, SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename PROFILE_DATA_TYPE, typename DISTANCE_TYPE, typename DATA_TYPE, typename SMEM_TYPE, typename Derived7Dist, typename Derived7, typename Derived4, typename Derived4f, typename Derived7uint>
__device__ inline FORCE_INLINE void do_row(
    SCAMPThreadInfo<DATA_TYPE> &info, Eigen::ArrayBase<Derived7Dist>& distc,
    const Eigen::ArrayBase<Derived7>& inormc, const Eigen::ArrayBase<Derived7>& dfc, const Eigen::ArrayBase<Derived7>& dgc,
    const Eigen::ArrayBase<Derived4>& inormr, const Eigen::ArrayBase<Derived4>& dfr, const Eigen::ArrayBase<Derived4>& dgr,
    const Eigen::ArrayBase<Derived4f>& curr_mp_row_val, Eigen::ArrayBase<Derived7uint>& idxc,
    SMEM_TYPE smem,
    OptionalArgs args) {
  // Compute the correlation values for the current tile row
  auto dist = (info.cov * inormc.segment<4>(iter) * inormr[iter]).template cast<DISTANCE_TYPE>();
  // Compute the next covariance values and update for the next iteration.
  info.cov += dfc.segment<4>(iter) * dgr[iter] + dgc.segment<4>(iter) * dfr[iter];

  // Update the column best-so-far values
  if (COMPUTE_COLS) {
    merge_to_column<iter, PROFILE_TYPE>(info, smem, distc, dist, idxc, args);
  }

  // Perform any updates for this tile row and commit to the shared-memory
  // matrix profile
  if (COMPUTE_ROWS) {
    update_row<iter, PROFILE_TYPE, DISTANCE_TYPE, PROFILE_DATA_TYPE>(
        info, smem, dist, curr_mp_row_val[iter], args);
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



template <typename SMEM_TYPE, typename DATA_TYPE,
          typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
void __device__
do_iteration_fast(SCAMPThreadInfo<DATA_TYPE> &info,
                  SMEM_TYPE &smem,
                  OptionalArgs &args) {
  // Arrays to store thread local variables, 7 columns, 4 rows
  Eigen::Array<DATA_TYPE, 7, 1> dfc, dgc, inormc;
  Eigen::Array4<DATA_TYPE> dgr, dfr, inormr;
  Eigen::Array4<float> mp_row_check;
  DISTANCE_TYPE init = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  Eigen::Array<DISTANCE_TYPE, 7, 1> distc = Eigen::Array<DISTANCE_TYPE, 7, 1>::Constant(init);
  Eigen::Array<unsigned int, 7 , 1> idxc;

  dfc = smem.df_col.segment<7>(info.local_col);
  dgc = smem.dg_col.segment<7>(info.local_col);
  inormc = smem.inorm_col.segment<7>(info.local_col);

  dfr = smem.df_row.segment<4>(info.local_row);
  dgr = smem.dg_row.segment<4>(info.local_row);
  inormr = smem.inorm_row.segment<4>(info.local_row);

  // For NN profiles we need to do a vectorized load to pull the best-so-far
  // values from cache
  if constexpr (COMPUTE_ROWS) {
    if constexpr (PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS || PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY || PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX) {
      Eigen::Array4<PROFILE_DATA_TYPE> temp = smem.local_mp_row.segment<4>(info.local_row);
      for (int i = 0; i < 4; ++i) {
        mp_entry e;
        e.ulong = temp[i];
        mp_row_check[i] = e.floats[0];
        break;
      }
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
      mp_row_check = smem.local_mp_row.segment<4>(info.local_row);
    }
  }

  // Generate and coalesce distances into profile
  do_row<0, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_DATA_TYPE, DISTANCE_TYPE>(info, distc, inormc, dfc, dgc, inormr, dfr, dgr, mp_row_check, idxc, smem, args);
  do_row<1, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_DATA_TYPE, DISTANCE_TYPE>(info, distc, inormc, dfc, dgc, inormr, dfr, dgr, mp_row_check, idxc, smem, args);
  do_row<2, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_DATA_TYPE, DISTANCE_TYPE>(info, distc, inormc, dfc, dgc, inormr, dfr, dgr, mp_row_check, idxc, smem, args);
  do_row<3, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_DATA_TYPE, DISTANCE_TYPE>(info, distc, inormc, dfc, dgc, inormr, dfr, dgr, mp_row_check, idxc, smem, args);

  // Update the column wise matrix profile with the best-so-far
  if (COMPUTE_COLS) {
    update_cols<PROFILE_TYPE, PROFILE_DATA_TYPE>(
        info, smem, distc, idxc);
  }

  // Advance counters
  info.local_col += DIAGS_PER_THREAD;
  info.local_row += DIAGS_PER_THREAD;
  info.global_col += DIAGS_PER_THREAD;
  info.global_row += DIAGS_PER_THREAD;
}

/////////////////////////////////////////////////////////////////////////
//  EDGE COMPUTATION
//////////////////////////////////////////////////////////////////////

template <SCAMPProfileType PROFILE_TYPE, typename PROFILE_DATA_TYPE, typename DerivedDist, typename DerivedSmemType>
__device__ inline void reduce_row(
    DerivedSmemType smem,
    int row, DerivedDist dist_row, uint32_t idx_row) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    fAtomicMax<ATOMIC_BLOCK>((float *)(smem.local_mp_row.data() + row), dist_row);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX || PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY || PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    MPatomicMax<ATOMIC_BLOCK>((uint64_t *)(smem.local_mp_row.data() + row), dist_row,
                            idx_row);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(smem.local_mp_row.data() + row,
                                                dist_row);
  } else {
    // Unimplemented.
  }
}

template <int iter, SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename PROFILE_DATA_TYPE,
          typename DerivedSmemType, typename DerivedDataType, typename DerivedDist, typename DerivedDist4>
__device__ inline void reduce_edge(
    DerivedSmemType smem,
    SCAMPThreadInfo<DerivedDataType> &info, Eigen::ArrayBase<DerivedDist4>& dist,
    DerivedDist &dist_row, uint32_t &idx_row, int diag, int num_diags, int n,
    OptionalArgs &args) {
  // Check if it is safe to do this iteration (we may have gone over the edge)
  if (info.global_col + iter < n && diag + iter < num_diags) {
    if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
      if (!isnan(dist[iter])) {
        if constexpr (COMPUTE_ROWS) {
          dist_row = fmaxf(dist_row, dist[iter]);
        }
        if constexpr (COMPUTE_COLS) {
          fAtomicMax<ATOMIC_BLOCK>(
              (float *)(smem.local_mp_col.data() + info.local_col + iter), dist[iter]);
        }
      }
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX || PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY || PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
      if constexpr (COMPUTE_ROWS) {
        if (dist[iter] > dist_row) {
          dist_row = dist[iter];
          idx_row = info.global_col + iter;
        }
      }
      if constexpr (COMPUTE_COLS) {
        MPatomicMax<ATOMIC_BLOCK>(
            (uint64_t *)(smem.local_mp_col.data() + info.local_col + iter), dist[iter],
            info.global_row);
      }
    } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
      if (dist[iter] > args.threshold) {
        if constexpr (COMPUTE_ROWS) {
          dist_row += dist[iter];
        }
        if constexpr (COMPUTE_COLS) {
          do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
              smem.local_mp_col.data() + info.local_col + iter, dist[iter]);
        }
      }
    } else {
      // Unimplemented.
    }

  }

}



template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE, typename DerivedSmemType, typename DerivedDataType>
__device__ inline void do_row_edge(
    SCAMPThreadInfo<DerivedDataType> &info,
    DerivedSmemType smem, int n,
    int diag, int num_diags, OptionalArgs &args) {
  DISTANCE_TYPE dist_row = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  auto distc = Eigen::Array4<DISTANCE_TYPE>::Constant(dist_row);
  Eigen::Array4<DISTANCE_TYPE> dist;
  int col = info.local_col;
  int row = info.local_row;
  uint32_t idx_row = 0;


  DerivedDataType inormr = smem.inorm_row[row];
  DerivedDataType dgr = smem.dg_row[row];
  DerivedDataType dfr = smem.df_row[row];

  // Compute the next set of distances (row y)
  dist = (info.cov * smem.inorm_col.segment<4>(col) * inormr).template cast<DISTANCE_TYPE>();

  // Update cov and compute the next distance values (row y)
  info.cov += smem.df_col.segment<4>(col) * dgr + smem.dg_col.segment<4>(col) * dfr;

  reduce_edge<0, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_DATA_TYPE>(smem, info, dist, dist_row, idx_row, diag, num_diags, n, args);
  reduce_edge<1, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_DATA_TYPE>(smem, info, dist, dist_row, idx_row, diag, num_diags, n, args);
  reduce_edge<2, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_DATA_TYPE>(smem, info, dist, dist_row, idx_row, diag, num_diags, n, args);
  reduce_edge<3, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_DATA_TYPE>(smem, info, dist, dist_row, idx_row, diag, num_diags, n, args);

  if (COMPUTE_ROWS) {
    reduce_row<PROFILE_TYPE, PROFILE_DATA_TYPE>(smem, row, dist_row,
                                                            idx_row);
  }
}
