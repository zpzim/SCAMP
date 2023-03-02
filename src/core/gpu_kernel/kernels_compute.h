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

template <SCAMPProfileType PROFILE_TYPE, typename DISTANCE_TYPE,
          typename InputDataType, typename DerivedSmem, typename DistRowArray>
__device__ inline void merge_to_row(const SCAMPKernelInputArgs<double>& args,
                                  const SCAMPThreadInfo<InputDataType, DISTANCE_TYPE>& info,
                                  DerivedSmem& smem,
                                  const Eigen::ArrayBase<DistRowArray>& dist,
                                  DISTANCE_TYPE& distr,
                                  unsigned int& idxr) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    DISTANCE_TYPE d = max_dist(dist);
    distr = fmaxf(distr, d);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    int idx;
    // TODO(fixme): Index calculation is wrong for warp edge
    DISTANCE_TYPE d = max_dist(dist, idx);
    idx += info.global_col;
    if (d > distr) {
      distr = d;
      idxr = idx;
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    DISTANCE_TYPE sum = (dist > args.opt.threshold).select(dist, 0).sum();
    distr += sum;
  } else {
    static_assert(PROFILE_TYPE != -1,
                  "update_row not implemented for profile type.");
  }
}

template <SCAMPProfileType PROFILE_TYPE, typename DerivedDistanceType,
          typename InputDataType, typename DerivedSmem>
__device__ inline void update_rows(const SCAMPKernelInputArgs<double>& args,
                                  const SCAMPThreadInfo<InputDataType, DerivedDistanceType>& info,
                                  DerivedSmem& smem,
                                  const DerivedDistanceType& distr,
                                  const unsigned int& idxr, int row_iter) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    float mp_row_check = smem.local_mp_row[info.local_row + row_iter];
    fAtomicMax_check<ATOMIC_BLOCK>(
          smem.local_mp_row.data() + info.local_row + row_iter, distr, mp_row_check);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    float mp_row_check;
    mp_entry e;
    e.ulong = smem.local_mp_row[info.local_row + row_iter];
    mp_row_check = e.floats[0];
    MPatomicMax_check<ATOMIC_BLOCK>(
        smem.local_mp_row.data() + info.local_row + row_iter, distr, idxr, mp_row_check);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    DerivedDistanceType sum = distr;
    #pragma unroll
    for (int i = 16; i >= 1; i /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    if ((threadIdx.x & 0x1f) == 0) {
      do_atomicAdd<DerivedDistanceType, ATOMIC_BLOCK>(
          smem.local_mp_row.data() + info.local_row + row_iter, sum);
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

template <SCAMPProfileType PROFILE_TYPE, typename DerivedDataType, typename DerivedDistanceType,
          typename DerivedSmem, typename ColDistArray, typename RowDistArray,
          typename ColIndexArray>
__device__ inline void merge_to_column(
    const SCAMPKernelInputArgs<double>& args,
    const SCAMPThreadInfo<DerivedDataType, DerivedDistanceType>& info, DerivedSmem smem,
    Eigen::ArrayBase<ColDistArray>& best_so_far,
    const Eigen::ArrayBase<RowDistArray>& dists_to_merge,
    Eigen::ArrayBase<ColIndexArray>& best_so_far_index, int row_iter) {
  static_assert(RowDistArray::RowsAtCompileTime == unrolled_diags);
  static_assert(ColDistArray::RowsAtCompileTime == ColIndexArray::RowsAtCompileTime);
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    #pragma unroll unrolled_diags
    for (int i = 0; i < unrolled_diags; ++i) {
      if (dists_to_merge[i] > best_so_far[i]) {
        best_so_far[i] = dists_to_merge[i];
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    #pragma unroll unrolled_diags
    for (int i = 0; i < unrolled_diags; ++i) {
      if (dists_to_merge[i] > best_so_far[i]) {
        best_so_far[i] = dists_to_merge[i];
        best_so_far_index[i] = info.global_row + row_iter;
      }
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    best_so_far += (dists_to_merge > args.opt.threshold).select(dists_to_merge, 0);
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
          typename DerivedSmemType, typename DerivedDistanceType>
__device__ inline void update_cols(const SCAMPKernelInputArgs<double>& args,
                                   SCAMPThreadInfo<DerivedInputDataType, DerivedDistanceType>& info,
                                   DerivedSmemType& smem,
                                   const DerivedDistanceType& distc,
                                   const unsigned int& idxc, int index) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    float mp_col_check = smem.local_mp_col[info.local_col + index];
    // Check the best-so-far column and update distance if necessary
    float old = fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col.data() + info.local_col + index, distc, mp_col_check);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    float mp_col_check;
    {
      mp_entry e;
      e.ulong = smem.local_mp_col[info.local_col + index];
      mp_col_check = e.floats[0];
    }

    // Check the best-so-far column and update distance/index if necessary
    MPatomicMax_check<ATOMIC_BLOCK>(
        smem.local_mp_col.data() + info.local_col + index, distc, idxc, mp_col_check);

  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    // Add the current sum that this thread has computed to the shared sum across
    // the entire thread block
    do_atomicAdd<double, ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + index, distc);

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
template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, typename DerivedDistanceType, typename DerivedInputType,
          typename DerivedSmem>
__device__ inline FORCE_INLINE void do_row(
    const SCAMPKernelInputArgs<double>& args, SCAMPThreadInfo<DerivedInputType, DerivedDistanceType>& info,
    DerivedSmem& smem, DerivedDistanceType& distr,
    const DerivedInputType& inormr,
    const DerivedInputType& dfr,
    const DerivedInputType& dgr,
    unsigned int& idxr, int row_iter) {

  // Compute the correlation values for the current tile row
  Eigen::Array<DerivedDistanceType, unrolled_diags, 1> dist;
  #pragma unroll unrolled_diags
  for (int i = 0; i < unrolled_diags; ++i) {
    dist[i] = info.cov[i] * info.inormc[i] * inormr;
    info.cov[i] = info.cov[i] + info.dfc[i] * dgr + info.dgc[i] * dfr;
  }
  

  // Update the column best-so-far values
  if constexpr (COMPUTE_COLS) {
    merge_to_column<PROFILE_TYPE>(args, info, smem, info.distc, dist, info.idxc, row_iter);
  }

  // Perform any updates for this tile row and commit to the shared-memory
  // matrix profile
  if constexpr (COMPUTE_ROWS) {
    merge_to_row<PROFILE_TYPE, DerivedDistanceType>(args, info, smem, dist, distr, idxr);
  }
}


template<typename Data, int size, typename T> 
auto __device__ inline ConvertToIntrinsic(const Eigen::ArrayBase<T>& arr) {
  static_assert(sizeof(Data) == 4 || sizeof(Data) == 8);
  static_assert(size > 0 && size <= 4); 
  if constexpr (size == 1) {
    return arr.coeff(0);
  }
  if constexpr (size == 2) {
    if constexpr (sizeof(Data) == 4) {
      float2 result;
      result.x = arr.coeff(0);
      result.y = arr.coeff(1);
      return result;
    }
    if constexpr (sizeof(Data) == 8) {
      double2 result;
      result.x = arr.coeff(0);
      result.y = arr.coeff(1);
      return result;
    }
  }
  if constexpr (size == 3) {
    if constexpr (sizeof(Data) == 4) {
      float3 result;
      result.x = arr.coeff(0);
      result.y = arr.coeff(1);
      result.z = arr.coeff(2);
      return result;
    }
    if constexpr (sizeof(Data) == 8) {
      double3 result;
      result.x = arr.coeff(0);
      result.y = arr.coeff(1);
      result.z = arr.coeff(2);
      return result;
    }
  }
  if constexpr (size == 4) {
    if constexpr (sizeof(Data) == 4) {
      float4 result;
      result.x = arr.coeff(0);
      result.y = arr.coeff(1);
      result.z = arr.coeff(2);
      result.w = arr.coeff(3);
      return result;
    }
    if constexpr (sizeof(Data) == 8) {
      double4 result;
      result.x = arr.coeff(0);
      result.y = arr.coeff(1);
      result.z = arr.coeff(2);
      result.w = arr.coeff(3);
      return result;
    }

  }
  static_assert(size <= 4);
}

template<typename Data, int size, typename T> 
Eigen::Array<Data, size, 1> __device__ inline ConvertToEigen(const T& intrinsic) {
  static_assert(sizeof(Data) == 4 || sizeof(Data) == 8);
  static_assert(size > 0 && size <= 4); 
  Eigen::Array<Data, size, 1> arr;
  if constexpr (size > 0) {
    arr[0] = intrinsic.x;
  }
  if constexpr (size > 1) {
    arr[1] = intrinsic.y;
  }
  if constexpr (size > 2) {
    arr[2] = intrinsic.z;
  }
  if constexpr (size > 3) {
    arr[3] = intrinsic.w;
  }
  return arr;  
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
          typename DerivedDistanceType, typename DerivedDataType,
          typename DerivedSmem>
void __device__ do_iteration_fast(const SCAMPKernelInputArgs<double>& args,
                                  SCAMPThreadInfo<DerivedDataType, DerivedDistanceType>& info,
                                  DerivedSmem& smem) {
  
  static_assert(unrolled_diags == DIAGS_PER_THREAD);
  constexpr int self_overlap = inner_unrolled_cols - unrolled_rows;

  cg::thread_block g = cg::this_thread_block();
  auto this_warp = cg::tiled_partition<32>(g);

  if (info.global_row == 0) {
    info.dfc = Eigen::Map<const Eigen::Array<double, unrolled_diags, 1>>(args.dfa + info.global_col).template cast<DerivedDataType>();
    info.dgc = Eigen::Map<const Eigen::Array<double, unrolled_diags, 1>>(args.dga + info.global_col).template cast<DerivedDataType>();
    info.inormc = Eigen::Map<const Eigen::Array<double, unrolled_diags, 1>>(args.normsa + info.global_col).template cast<DerivedDataType>();
  }

  for (int j = 0; j < outer_unrolled_rows; ++j) {
    DerivedDataType dfr = smem.df_row[info.local_row + j];
    DerivedDataType dgr = smem.dg_row[info.local_row + j];
    DerivedDataType inormr = smem.inorm_row[info.local_row + j];
    DerivedDistanceType distr = init_dist<DerivedDistanceType, PROFILE_TYPE>();
    unsigned int idxr;
    do_row<PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS>(
        args, info, smem, distr, inormr, dfr, dgr, idxr, j);
    DerivedDataType temp = this_warp.shfl(info.cov[unrolled_diags - 1], info.srcln); 
    for (int i = 0; i < unrolled_diags - 1; ++i) {
      info.cov[i + 1] = info.cov[i]; 
    }
    info.cov[0] = temp;
    if (info.updates_remaining < unrolled_diags) {
      int new_global_col = info.global_col + 32 * unrolled_diags;
      int new_local_col = info.local_col + 32 * unrolled_diags;
      int col_to_update = (unrolled_diags - 1) - info.updates_remaining;
      // Update the column wise matrix profile with the best-so-far
      if constexpr (COMPUTE_COLS) {
        update_cols<PROFILE_TYPE>(args, info, smem, info.distc[col_to_update], info.idxc[col_to_update], col_to_update);
      }
      info.dfc[col_to_update] = args.dfa[new_global_col + col_to_update];
      info.dgc[col_to_update] = args.dga[new_global_col + col_to_update];
      info.inormc[col_to_update] = args.normsa[new_global_col + col_to_update];
      info.distc[col_to_update] = init_dist<DerivedDistanceType, PROFILE_TYPE>();
      if (info.updates_remaining == 0) {
        info.global_col = new_global_col;
        info.local_col = new_local_col;
        info.updates_remaining = 32 * unrolled_diags;
      }
    }
    info.updates_remaining--;
    // Update the row wise matrix profile with the best-so-far
    if constexpr (COMPUTE_ROWS) {
      update_rows<PROFILE_TYPE>(args, info, smem, distr, idxr, j);
    }
  };

  // Update the column wise matrix profile with the remaining best-so-far.
  //if constexpr (COMPUTE_COLS) {
  //  update_cols<PROFILE_TYPE>(args, info, smem, distc, idxc[i], );
  //}
  // Advance counters
  info.local_row += outer_unrolled_rows;
  info.global_row += outer_unrolled_rows;
}

/////////////////////////////////////////////////////////////////////////
//  EDGE COMPUTATION
//////////////////////////////////////////////////////////////////////

template <SCAMPProfileType PROFILE_TYPE, typename DerivedDataType,
          typename DerivedDist, typename DerivedSmemType>
__device__ inline void reduce_row(const SCAMPKernelInputArgs<double>& args,
                                  const SCAMPThreadInfo<DerivedDataType, DerivedDist>& info,
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
                                   const SCAMPThreadInfo<DerivedDataType, DerivedDist>& info,
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
          typename DerivedDistanceType, typename DerivedSmemType,
          typename DerivedDataType>
__device__ inline void do_row_edge(const SCAMPKernelInputArgs<double>& args,
                                   SCAMPThreadInfo<DerivedDataType, DerivedDistanceType>& info,
                                   DerivedSmemType& smem, int diag,
                                   int num_diags) {
  DerivedDistanceType dist_row = init_dist<DerivedDistanceType, PROFILE_TYPE>();
  Eigen::Array<DerivedDistanceType, DIAGS_PER_THREAD, 1> distc =
      Eigen::Array<DerivedDistanceType, DIAGS_PER_THREAD, 1>::Constant(dist_row);
  uint32_t idx_row = 0;
  DerivedDataType inormr = smem.inorm_row[info.local_row];
  DerivedDataType dgr = smem.dg_row[info.local_row];
  DerivedDataType dfr = smem.df_row[info.local_row];

  Eigen::Array<DerivedDataType,DIAGS_PER_THREAD, 1> dfc, dgc, inormc;
  dfc = Eigen::Map<const Eigen::Array<double, DIAGS_PER_THREAD, 1>>(args.dfa + info.global_col).template cast<DerivedDataType>();
  dgc = Eigen::Map<const Eigen::Array<double, DIAGS_PER_THREAD, 1>>(args.dga + info.global_col).template cast<DerivedDataType>();
  inormc = Eigen::Map<const Eigen::Array<double, DIAGS_PER_THREAD, 1>>(args.normsa + info.global_col).template cast<DerivedDataType>();

  // Compute the next set of distances. Only a single row. Note this may compute
  // garbage for values beyond the edge of the array.
  Eigen::Array<DerivedDistanceType, DIAGS_PER_THREAD, 1> dist =
      (info.cov * inormc * inormr)
          .template cast<DerivedDistanceType>();

  // Update cov and compute the next distance values. Note this may compute
  // garbage for values beyond the edge of the array.
  info.cov += dfc * dgr +
              dgc * dfr;

  for_<DIAGS_PER_THREAD>([&](auto i) {
    reduce_edge<i.value, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS>(
        args, info, smem, dist, dist_row, idx_row, diag, num_diags);
  });

  if constexpr (COMPUTE_ROWS) {
    reduce_row<PROFILE_TYPE>(args, info, smem, dist_row, idx_row);
  }
}
