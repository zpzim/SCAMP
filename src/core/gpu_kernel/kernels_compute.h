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

static __inline__ __device__ double fetch_double(uint2 p) {
  return __hiloint2double(p.y, p.x);
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

template <int row_iter, SCAMPProfileType PROFILE_TYPE, typename DISTANCE_TYPE,
          typename InputDataType, typename DerivedSmem, typename DistRowArray>
__device__ inline void merge_to_row(const SCAMPKernelInputArgs<double>& args,
                                    const SCAMPThreadInfo<InputDataType>& info,
                                    DerivedSmem& smem,
                                    const Eigen::ArrayBase<DistRowArray>& dist,
                                    DISTANCE_TYPE& distr, unsigned int& idxr) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    DISTANCE_TYPE d = max_dist(dist);
    distr = fmaxf(distr, d);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    int idx;
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
    static_assert(PROFILE_TYPE != -1,
                  "update_row not implemented for profile type.");
  }
}

template <int row_iter, int num_to_update, SCAMPProfileType PROFILE_TYPE,
          typename DISTANCE_TYPE, typename InputDataType, typename DerivedSmem,
          typename DistRowArray, typename IndexRowArray>
__device__ inline void update_rows(
    const SCAMPKernelInputArgs<double>& args,
    const SCAMPThreadInfo<InputDataType>& info, DerivedSmem& smem,
    const Eigen::ArrayBase<DistRowArray>& distr,
    const Eigen::ArrayBase<IndexRowArray>& idxr) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    Eigen::Array<float, num_to_update, 1> mp_row_check =
        smem.local_mp_row.segment<num_to_update>(info.local_row + row_iter);
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
        smem.local_mp_row.segment<num_to_update>(info.local_row + row_iter);
#pragma unroll num_to_update
    for (int i = 0; i < num_to_update; ++i) {
      mp_entry e;
      e.ulong = temp[i];
      mp_row_check[i] = e.floats[0];
    }
    for (int r = 0; r < num_to_update; ++r) {
      MPatomicMax_check<ATOMIC_BLOCK>(
          smem.local_mp_row.data() + info.local_row + r + row_iter,
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

template <int row_iter, SCAMPProfileType PROFILE_TYPE, typename DerivedDataType,
          typename DerivedSmem, typename ColDistArray, typename RowDistArray,
          typename ColIndexArray>
__device__ inline void merge_to_column(
    const SCAMPKernelInputArgs<double>& args,
    const SCAMPThreadInfo<DerivedDataType>& info, DerivedSmem smem,
    Eigen::ArrayBase<ColDistArray>& best_so_far,
    const Eigen::ArrayBase<RowDistArray>& dists_to_merge,
    Eigen::ArrayBase<ColIndexArray>& best_so_far_index) {
  static_assert(RowDistArray::RowsAtCompileTime == unrolled_diags);
  static_assert(ColDistArray::RowsAtCompileTime ==
                ColIndexArray::RowsAtCompileTime);
  static_assert(ColDistArray::RowsAtCompileTime == unrolled_cols);
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
    best_so_far.segment<unrolled_diags>(row_iter) +=
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

template <int start_index, int num_to_update, SCAMPProfileType PROFILE_TYPE,
          typename DerivedInputDataType, typename DerivedSmemType,
          typename ColDistArray, typename ColIndexArray>
__device__ inline void update_cols(const SCAMPKernelInputArgs<double>& args,
                                   SCAMPThreadInfo<DerivedInputDataType>& info,
                                   DerivedSmemType& smem,
                                   Eigen::ArrayBase<ColDistArray>& distc,
                                   Eigen::ArrayBase<ColIndexArray>& idxc) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    Eigen::Array<float, num_to_update, 1> mp_col_check =
        smem.local_mp_col.segment<num_to_update>(info.local_col + start_index);
// Check the best-so-far column and update distance if necessary
#pragma unroll num_to_update
    for (int i = 0; i < num_to_update; ++i) {
      float old = fAtomicMax_check<ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + i + start_index,
          distc[i + start_index], mp_col_check[i]);
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                       PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY ||
                       PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    Eigen::Array<float, num_to_update, 1> mp_col_check;
    {
      Eigen::Array<uint64_t, num_to_update, 1> temp =
          smem.local_mp_col.segment<num_to_update>(info.local_col +
                                                   start_index);
#pragma unroll num_to_update
      for (int i = 0; i < num_to_update; ++i) {
        mp_entry e;
        e.ulong = temp[i];
        mp_col_check[i] = e.floats[0];
      }
    }

// Check the best-so-far column and update distance/index if necessary
#pragma unroll num_to_update
    for (int i = 0; i < num_to_update; ++i) {
      MPatomicMax_check<ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + i + start_index,
          distc[i + start_index], idxc[i + start_index], mp_col_check[i]);
    }

  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
// Add the current sum that this thread has computed to the shared sum across
// the entire thread block
#pragma unroll num_to_update
    for (int i = 0; i < num_to_update; ++i) {
      do_atomicAdd<double, ATOMIC_BLOCK>(
          smem.local_mp_col.data() + info.local_col + i + start_index,
          distc[i + start_index]);
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
template <int outer_row_iter, int row_iter, SCAMPProfileType PROFILE_TYPE,
          bool COMPUTE_ROWS, bool COMPUTE_COLS, typename DISTANCE_TYPE,
          typename DerivedInputType, typename DerivedSmem,
          typename DistColArray, typename IndexColArray>
__device__ inline FORCE_INLINE void do_row(
    const SCAMPKernelInputArgs<double>& args,
    SCAMPThreadInfo<DerivedInputType>& info, DerivedSmem& smem,
    Eigen::ArrayBase<DistColArray>& distc, DISTANCE_TYPE& distr,
    const DerivedInputType& inormr, const DerivedInputType& dfr,
    const DerivedInputType& dgr, Eigen::ArrayBase<IndexColArray>& idxc,
    unsigned int& idxr) {
  // Compute the correlation values for the current tile row
  Eigen::Array<DISTANCE_TYPE, unrolled_diags, 1> dist;
#pragma unroll unrolled_diags
  for (int i = 0; i < unrolled_diags; ++i) {
    dist[i] = info.cov[i] * info.inormc[row_iter + i] * inormr;
    info.cov[i] = info.cov[i] + info.dfc[row_iter + i] * dgr +
                  info.dgc[row_iter + i] * dfr;
  }

  // Update the column best-so-far values
  if constexpr (COMPUTE_COLS) {
    merge_to_column<outer_row_iter, PROFILE_TYPE>(args, info, smem, distc, dist,
                                                  idxc);
  }

  // Perform any updates for this tile row and commit to the shared-memory
  // matrix profile
  if constexpr (COMPUTE_ROWS) {
    merge_to_row<outer_row_iter, PROFILE_TYPE, DISTANCE_TYPE>(
        args, info, smem, dist, distr, idxr);
  }
}

template <typename Data, int size, typename T>
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

template <typename Data, int size, typename T>
Eigen::Array<Data, size, 1> __device__ inline ConvertToEigen(
    const T& intrinsic) {
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
          typename DISTANCE_TYPE, typename DerivedDataType,
          typename DerivedSmem>
void __device__ do_iteration_fast(const SCAMPKernelInputArgs<double>& args,
                                  SCAMPThreadInfo<DerivedDataType>& info,
                                  DerivedSmem& smem) {
  DISTANCE_TYPE init = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  Eigen::Array<DISTANCE_TYPE, unrolled_cols, 1> distc =
      Eigen::Array<DISTANCE_TYPE, unrolled_cols, 1>::Constant(init);
  Eigen::Array<DISTANCE_TYPE, outer_unrolled_rows, 1> distr =
      Eigen::Array<DISTANCE_TYPE, outer_unrolled_rows, 1>::Constant(init);
  Eigen::Array<unsigned int, unrolled_cols, 1> idxc;
  Eigen::Array<unsigned int, outer_unrolled_rows, 1> idxr;
  static_assert(unrolled_diags == DIAGS_PER_THREAD);

  constexpr int self_overlap = inner_unrolled_cols - unrolled_rows;

  cg::thread_block g = cg::this_thread_block();
  auto this_warp = cg::tiled_partition<32>(g);

  if (info.global_row == 0) {
    for (int i = 0; i < inner_unrolled_cols; ++i) {
      info.dfc[i] =
          fetch_double(tex1Dfetch<uint2>(args.dfa_tex, info.global_col + i));
      info.dgc[i] =
          fetch_double(tex1Dfetch<uint2>(args.dga_tex, info.global_col + i));
      info.inormc[i] =
          fetch_double(tex1Dfetch<uint2>(args.normsa_tex, info.global_col + i));
      // info.dfc[i] = args.dfa[info.global_col + i];
      // info.dgc[i] = args.dga[info.global_col + i];
      // info.inormc[i] = args.normsa[info.global_col +i];
    }
    // info.dfc = Eigen::Map<const Eigen::Array<double, inner_unrolled_cols,
    // 1>>(args.dfa + info.global_col).template cast<DerivedDataType>(); info.dgc
    // = Eigen::Map<const Eigen::Array<double, inner_unrolled_cols, 1>>(args.dga
    // + info.global_col).template cast<DerivedDataType>(); info.inormc =
    // Eigen::Map<const Eigen::Array<double, inner_unrolled_cols, 1>>(args.normsa
    // + info.global_col).template cast<DerivedDataType>();
  }

  for_<outer_unrolled_rows / unrolled_rows>([&](auto j) {
    if (info.global_row != 0 || j.value > 0) {
      /*
            auto temp = ConvertToIntrinsic<DerivedDataType,
         unrolled_rows>(info.dfc.segment<unrolled_rows>(self_overlap));
            info.dfc.segment<self_overlap>(0) =
         info.dfc.segment<self_overlap>(unrolled_rows); temp =
         this_warp.shfl_down(temp,1);
            info.dfc.segment<unrolled_rows>(self_overlap) =
         ConvertToEigen<DerivedDataType, unrolled_rows>(temp);

            temp = ConvertToIntrinsic<DerivedDataType,
         unrolled_rows>(info.dgc.segment<unrolled_rows>(self_overlap));
            info.dgc.segment<self_overlap>(0) =
         info.dgc.segment<self_overlap>(unrolled_rows); temp =
         this_warp.shfl_down(temp,1);
            info.dgc.segment<unrolled_rows>(self_overlap) =
         ConvertToEigen<DerivedDataType, unrolled_rows>(temp);

            temp = ConvertToIntrinsic<DerivedDataType,
         unrolled_rows>(info.inormc.segment<unrolled_rows>(self_overlap));
            info.inormc.segment<self_overlap>(0) =
         info.inormc.segment<self_overlap>(unrolled_rows); temp =
         this_warp.shfl_down(temp,1);
            info.inormc.segment<unrolled_rows>(self_overlap) =
         ConvertToEigen<DerivedDataType, unrolled_rows>(temp);
      */
      info.dfc.segment<self_overlap>(0) =
          info.dfc.segment<self_overlap>(unrolled_rows);
      info.dgc.segment<self_overlap>(0) =
          info.dgc.segment<self_overlap>(unrolled_rows);
      info.inormc.segment<self_overlap>(0) =
          info.inormc.segment<self_overlap>(unrolled_rows);

      // if(this_warp.thread_rank() == 31) {
      for (int i = 0; i < unrolled_rows; ++i) {
        info.dfc[self_overlap + i] = fetch_double(tex1Dfetch<uint2>(
            args.dfa_tex,
            info.global_col + j.value * unrolled_rows + self_overlap + i));
        info.dgc[self_overlap + i] = fetch_double(tex1Dfetch<uint2>(
            args.dga_tex,
            info.global_col + j.value * unrolled_rows + self_overlap + i));
        info.inormc[self_overlap + i] = fetch_double(tex1Dfetch<uint2>(
            args.normsa_tex,
            info.global_col + j.value * unrolled_rows + self_overlap + i));
        // info.dfc[self_overlap+i] = args.dfa[info.global_col + j.value *
        // unrolled_rows + self_overlap+i]; info.dgc[self_overlap+i] =
        // args.dga[info.global_col + j.value * unrolled_rows + self_overlap+i];
        // info.inormc[self_overlap+i] = args.normsa[info.global_col + j.value *
        // unrolled_rows + self_overlap+i];

        // info.dfc.segment<unrolled_rows>(self_overlap) = Eigen::Map<const
        // Eigen::Array<double, unrolled_rows, 1>>(args.dfa + info.global_col +
        // j.value * unrolled_rows + self_overlap).template
        // cast<DerivedDataType>(); info.dgc.segment<unrolled_rows>(self_overlap)
        // = Eigen::Map<const Eigen::Array<double, unrolled_rows, 1>>(args.dga +
        // info.global_col + j.value * unrolled_rows + self_overlap).template
        // cast<DerivedDataType>();
        // info.inormc.segment<unrolled_rows>(self_overlap) = Eigen::Map<const
        // Eigen::Array<double, unrolled_rows, 1>>(args.normsa + info.global_col
        // + j.value * unrolled_rows + self_overlap).template
        // cast<DerivedDataType>();
      }
      // info.dfc.segment<unrolled_rows>(self_overlap) = Eigen::Map<const
      // Eigen::Array<double, unrolled_rows, 1>>(args.dfa + info.global_col +
      // j.value * unrolled_rows + self_overlap).template
      // cast<DerivedDataType>(); info.dgc.segment<unrolled_rows>(self_overlap) =
      // Eigen::Map<const Eigen::Array<double, unrolled_rows, 1>>(args.dga +
      // info.global_col + j.value * unrolled_rows + self_overlap).template
      // cast<DerivedDataType>();
      // info.inormc.segment<unrolled_rows>(self_overlap) = Eigen::Map<const
      // Eigen::Array<double, unrolled_rows, 1>>(args.normsa + info.global_col +
      // j.value * unrolled_rows + self_overlap).template
      // cast<DerivedDataType>();
      //}
    }

    Eigen::Array<DerivedDataType, unrolled_rows, 1> dfr =
        smem.df_row.segment<unrolled_rows>(info.local_row +
                                           j.value * unrolled_rows);
    Eigen::Array<DerivedDataType, unrolled_rows, 1> dgr =
        smem.dg_row.segment<unrolled_rows>(info.local_row +
                                           j.value * unrolled_rows);
    Eigen::Array<DerivedDataType, unrolled_rows, 1> inormr =
        smem.inorm_row.segment<unrolled_rows>(info.local_row +
                                              j.value * unrolled_rows);
    for_<unrolled_rows>([&](auto k) {
      do_row<j.value * unrolled_rows + k.value, k.value, PROFILE_TYPE,
             COMPUTE_ROWS, COMPUTE_COLS, DISTANCE_TYPE>(
          args, info, smem, distc, distr[j.value * unrolled_rows + k.value],
          inormr[k.value], dfr[k.value], dgr[k.value], idxc,
          idxr[j.value * unrolled_rows + k.value]);
    });
    // Update the column wise matrix profile with the best-so-far
    if constexpr (COMPUTE_COLS) {
      update_cols<j.value * unrolled_rows, unrolled_rows, PROFILE_TYPE>(
          args, info, smem, distc, idxc);
    }
    // Update the row wise matrix profile with the best-so-far
    if constexpr (COMPUTE_ROWS) {
      update_rows<j.value * unrolled_rows, unrolled_rows, PROFILE_TYPE,
                  DISTANCE_TYPE>(args, info, smem, distr, idxr);
    }
  });

  // Update the column wise matrix profile with the remaining best-so-far.
  if constexpr (COMPUTE_COLS) {
    update_cols<outer_unrolled_rows, unrolled_cols - outer_unrolled_rows,
                PROFILE_TYPE>(args, info, smem, distc, idxc);
  }
  // Advance counters
  info.local_col += outer_unrolled_rows;
  info.local_row += outer_unrolled_rows;
  info.global_col += outer_unrolled_rows;
  info.global_row += outer_unrolled_rows;
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

  Eigen::Array<DerivedDataType, DIAGS_PER_THREAD, 1> dfc, dgc, inormc;
  dfc = Eigen::Map<const Eigen::Array<double, DIAGS_PER_THREAD, 1>>(
            args.dfa + info.global_col)
            .template cast<DerivedDataType>();
  dgc = Eigen::Map<const Eigen::Array<double, DIAGS_PER_THREAD, 1>>(
            args.dga + info.global_col)
            .template cast<DerivedDataType>();
  inormc = Eigen::Map<const Eigen::Array<double, DIAGS_PER_THREAD, 1>>(
               args.normsa + info.global_col)
               .template cast<DerivedDataType>();

  // Compute the next set of distances. Only a single row. Note this may compute
  // garbage for values beyond the edge of the array.
  Eigen::Array<DISTANCE_TYPE, DIAGS_PER_THREAD, 1> dist =
      (info.cov * inormc * inormr).template cast<DISTANCE_TYPE>();

  // Update cov and compute the next distance values. Note this may compute
  // garbage for values beyond the edge of the array.
  info.cov += dfc * dgr + dgc * dfr;

  for_<DIAGS_PER_THREAD>([&](auto i) {
    reduce_edge<i.value, PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS>(
        args, info, smem, dist, dist_row, idx_row, diag, num_diags);
  });

  if constexpr (COMPUTE_ROWS) {
    reduce_row<PROFILE_TYPE>(args, info, smem, dist_row, idx_row);
  }
}
