#pragma once

#include "defines.h"

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

// UPDATE_ROW when PROFILE_TYPE == PROFILE_TYPE_1NN
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void update_row(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN> smem,
    const DISTANCE_TYPE dist[4], const float curr_mp_row_val,
    const OptionalArgs args) {
  DISTANCE_TYPE d = max4(dist[0], dist[1], dist[2], dist[3]);
  fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_row + info.local_row + iter, d,
                                 curr_mp_row_val);
}

// UPDATE_ROW when PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void update_row(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_APPROX_ALL_NEIGHBORS>
        smem,
    const DISTANCE_TYPE dist[4], const float curr_mp_row_val,
    const OptionalArgs args) {
  uint32_t idx;
  DISTANCE_TYPE d = max4_index<DISTANCE_TYPE>(
      dist[0], dist[1], dist[2], dist[3], info.global_col + iter, idx);
  MPatomicMax_check<ATOMIC_BLOCK>(
      (uint64_t *)(smem.local_mp_row + info.local_row + iter), d, idx,
      curr_mp_row_val);
}

// UPDATE_ROW when PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void update_row(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_MATRIX_SUMMARY> smem,
    const DISTANCE_TYPE dist[4], const float curr_mp_row_val,
    const OptionalArgs args) {
  uint32_t idx;
  DISTANCE_TYPE d = max4_index<DISTANCE_TYPE>(
      dist[0], dist[1], dist[2], dist[3], info.global_col + iter, idx);
  MPatomicMax_check<ATOMIC_BLOCK>(
      (uint64_t *)(smem.local_mp_row + info.local_row + iter), d, idx,
      curr_mp_row_val);
}

// UPDATE_ROW when PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void update_row(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN_INDEX> smem,
    const DISTANCE_TYPE dist[4], const float curr_mp_row_val,
    const OptionalArgs args) {
  uint32_t idx;
  DISTANCE_TYPE d = max4_index<DISTANCE_TYPE>(
      dist[0], dist[1], dist[2], dist[3], info.global_col + iter, idx);
  MPatomicMax_check<ATOMIC_BLOCK>(
      (uint64_t *)(smem.local_mp_row + info.local_row + iter), d, idx,
      curr_mp_row_val);
}

// UPDATE_ROW when PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void update_row(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_SUM_THRESH> smem,
    const DISTANCE_TYPE dist[4], const float curr_mp_row_val,
    const OptionalArgs args) {
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
        smem.local_mp_row + info.local_row + iter,
        static_cast<PROFILE_DATA_TYPE>(sum));
  }
}

/*
// TO ADD A NEW PROFILE TYPE IMPLEMENT THIS FUNCTION FOR THAT TYPE

// UPDATE_ROW when PROFILE_TYPE == PROFILE_TYPE_???
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void update_row(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
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

// MERGE_TO_COLUMN when PROFILE_TYPE == PROFILE_TYPE_1NN
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void merge_to_column(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    const SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN> smem,
    DISTANCE_TYPE best_so_far[7], const DISTANCE_TYPE dists_to_merge[4],
    unsigned int best_so_far_index[7], const OptionalArgs args) {
#pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    if (dists_to_merge[i] > best_so_far[iter + i]) {
      best_so_far[iter + i] = dists_to_merge[i];
    }
  }
}

// MERGE_TO_COLUMN when PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void merge_to_column(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    const SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN_INDEX> smem,
    DISTANCE_TYPE best_so_far[7], const DISTANCE_TYPE dists_to_merge[4],
    unsigned int best_so_far_index[7], const OptionalArgs args) {
#pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    if (dists_to_merge[i] > best_so_far[iter + i]) {
      best_so_far[iter + i] = dists_to_merge[i];
      best_so_far_index[iter + i] = info.global_row + iter;
    }
  }
}

// MERGE_TO_COLUMN when PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void merge_to_column(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    const SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_MATRIX_SUMMARY>
        smem,
    DISTANCE_TYPE best_so_far[7], const DISTANCE_TYPE dists_to_merge[4],
    unsigned int best_so_far_index[7], const OptionalArgs args) {
#pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    if (dists_to_merge[i] > best_so_far[iter + i]) {
      best_so_far[iter + i] = dists_to_merge[i];
      best_so_far_index[iter + i] = info.global_row + iter;
    }
  }
}

// MERGE_TO_COLUMN when PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void merge_to_column(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    const SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE,
                    PROFILE_TYPE_APPROX_ALL_NEIGHBORS>
        smem,
    DISTANCE_TYPE best_so_far[7], const DISTANCE_TYPE dists_to_merge[4],
    unsigned int best_so_far_index[7], const OptionalArgs args) {
#pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    if (dists_to_merge[i] > best_so_far[iter + i]) {
      best_so_far[iter + i] = dists_to_merge[i];
      best_so_far_index[iter + i] = info.global_row + iter;
    }
  }
}

// MERGE_TO_COLUMN when PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void merge_to_column(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
    const SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_SUM_THRESH> smem,
    DISTANCE_TYPE best_so_far[7], const DISTANCE_TYPE dists_to_merge[4],
    unsigned int best_so_far_index[7], const OptionalArgs args) {
#pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    if (dists_to_merge[i] > args.threshold) {
      best_so_far[iter + i] += dists_to_merge[i];
    }
  }
}

/*
// TO ADD A NEW PROFILE TYPE IMPLEMENT THIS FUNCTION FOR THAT TYPE

// MERGE_TO_COLUMN when PROFILE_TYPE == PROFILE_TYPE_???
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE>
__device__ inline void merge_to_column(
    const SCAMPThreadInfo<ACCUM_TYPE> info,
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

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, SCAMPProfileType PROFILE_TYPE>
__device__ inline void update_cols_std(
    SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> smem,
    DISTANCE_TYPE distc[7], unsigned int idxc[7]) {
  int c = info.local_col >> 2;

  ulonglong4 mp_col_check1, mp_col_check2;
  float mp_col_check[7];

  // Load the best-so-far values 4 at a time to reduce shared memory
  // transactions We are preloading these values to recuce the number of atomic
  // operations in MPatomicMax_check this is a 'test-and-test-and-set' strategy
  mp_col_check1 = reinterpret_cast<ulonglong4 *>(smem.local_mp_col)[c];
  mp_col_check2 = reinterpret_cast<ulonglong4 *>(smem.local_mp_col)[c + 1];

  // Rename to array layout for loop
  mp_entry e;
  e.ulong = mp_col_check1.x;
  mp_col_check[0] = e.floats[0];
  e.ulong = mp_col_check1.y;
  mp_col_check[1] = e.floats[0];
  e.ulong = mp_col_check1.z;
  mp_col_check[2] = e.floats[0];
  e.ulong = mp_col_check1.w;
  mp_col_check[3] = e.floats[0];
  e.ulong = mp_col_check2.x;
  mp_col_check[4] = e.floats[0];
  e.ulong = mp_col_check2.y;
  mp_col_check[5] = e.floats[0];
  e.ulong = mp_col_check2.z;
  mp_col_check[6] = e.floats[0];

// Check the best-so-far column and update distance/index if necessary
#pragma unroll 7
  for (int i = 0; i < 7; ++i) {
    MPatomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col + info.local_col + i,
                                    distc[i], idxc[i], mp_col_check[i]);
  }
}

// UPDATE COLS where PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX
template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void update_cols(
    SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN_INDEX> smem,
    DISTANCE_TYPE distc[7], unsigned int idxc[7]) {
  update_cols_std<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                  PROFILE_TYPE_1NN_INDEX>(info, smem, distc, idxc);
}

// UPDATE COLS where PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY
template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void update_cols(
    SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_MATRIX_SUMMARY> smem,
    DISTANCE_TYPE distc[7], unsigned int idxc[7]) {
  update_cols_std<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                  PROFILE_TYPE_MATRIX_SUMMARY>(info, smem, distc, idxc);
}

// UPDATE COLS where PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS
template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void update_cols(
    SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_APPROX_ALL_NEIGHBORS>
        smem,
    DISTANCE_TYPE distc[7], unsigned int idxc[7]) {
  update_cols_std<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                  PROFILE_TYPE_APPROX_ALL_NEIGHBORS>(info, smem, distc, idxc);
}

// UPDATE_COLS where PROFILE_TYPE == PROFILE_TYPE_1NN
template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void update_cols(
    SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN> smem,
    DISTANCE_TYPE distc[7], unsigned int idxc[7]) {
  int c = info.local_col >> 2;
  float4 mp_col_check1, mp_col_check2;
  float mp_col_check[7];

  // Load the best-so-far values 4 at a time to reduce latency
  mp_col_check1 = reinterpret_cast<float4 *>(smem.local_mp_col)[c];
  mp_col_check2 = reinterpret_cast<float4 *>(smem.local_mp_col)[c + 1];

  // Rename to array layout for loop
  mp_col_check[0] = mp_col_check1.x;
  mp_col_check[1] = mp_col_check1.y;
  mp_col_check[2] = mp_col_check1.z;
  mp_col_check[3] = mp_col_check1.w;
  mp_col_check[4] = mp_col_check2.x;
  mp_col_check[5] = mp_col_check2.y;
  mp_col_check[6] = mp_col_check2.z;

// Check the best-so-far column and update distance if necessary
#pragma unroll 7
  for (int i = 0; i < 7; ++i) {
    fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col + info.local_col + i,
                                   distc[i], mp_col_check[i]);
  }
}

// UPDATE COLS where PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH
template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void update_cols(
    SCAMPThreadInfo<ACCUM_TYPE> info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_SUM_THRESH> smem,
    DISTANCE_TYPE distc[7], unsigned int idxc[7]) {
// Add the current sum that this thread has computed to the shared sum across
// the entire thread block
#pragma unroll 7
  for (int i = 0; i < 7; ++i) {
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
        smem.local_mp_col + info.local_col + i, distc[i]);
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
template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, SCAMPProfileType PROFILE_TYPE>
__device__ inline FORCE_INLINE void do_row(
    SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE distc[7],
    const DATA_TYPE inormc[7], const DATA_TYPE dfc[7], const DATA_TYPE dgc[7],
    const DATA_TYPE inormr[4], const DATA_TYPE dfr[4], const DATA_TYPE dgr[4],
    const float curr_mp_row_val[4], unsigned int idxc[7],
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> smem,
    OptionalArgs args) {
  DISTANCE_TYPE dist[4];

  // Compute the correlation values for the current tile row
  dist[0] = info.cov1 * inormc[iter] * inormr[iter];
  dist[1] = info.cov2 * inormc[iter + 1] * inormr[iter];
  dist[2] = info.cov3 * inormc[iter + 2] * inormr[iter];
  dist[3] = info.cov4 * inormc[iter + 3] * inormr[iter];

  // Compute the next covariance values and update our registers for the next
  // iteration
  info.cov1 = info.cov1 + dfc[iter] * dgr[iter] + dgc[iter] * dfr[iter];
  info.cov2 = info.cov2 + dfc[iter + 1] * dgr[iter] + dgc[iter + 1] * dfr[iter];
  info.cov3 = info.cov3 + dfc[iter + 2] * dgr[iter] + dgc[iter + 2] * dfr[iter];
  info.cov4 = info.cov4 + dfc[iter + 3] * dgr[iter] + dgc[iter + 3] * dfr[iter];

  // Perform any profile-specific distance calculations
  // Update the column best-so-far values
  if (COMPUTE_COLS) {
    merge_to_column<iter, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE,
                    DISTANCE_TYPE>(info, smem, distc, dist, idxc, args);
  }

  // Perform any updates for this tile row and commit to the shared-memory
  // matrix profile
  if (COMPUTE_ROWS) {
    update_row<iter, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE>(
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

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
void __device__
do_iteration_fast(SCAMPThreadInfo<ACCUM_TYPE> &info,
                  SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem,
                  OptionalArgs &args) {
  // Load row values 4 at a time, load column values 4 at a time
  int r = info.local_row >> 2;
  int c = info.local_col >> 2;

  // Arrays to store thread local variables, 7 columns, 4 rows
  DATA_TYPE dfc[7], dgc[7], inormc[7];
  DATA_TYPE dgr[4], dfr[4], inormr[4];
  float mp_row_check[4];
  DISTANCE_TYPE init = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  DISTANCE_TYPE distc[7] = {init, init, init, init, init, init, init};
  unsigned int idxc[7];

  // Preload the shared memory values we will use into registers using
  // vectorized loads
  VEC4_DATA_TYPE dfc_temp = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c];
  VEC4_DATA_TYPE dgc_temp = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c];
  VEC4_DATA_TYPE inormc_temp =
      (reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c]);

  // Rename to array structure for ease of use
  dfc[0] = dfc_temp.x;
  dfc[1] = dfc_temp.y;
  dfc[2] = dfc_temp.z;
  dfc[3] = dfc_temp.w;
  dgc[0] = dgc_temp.x;
  dgc[1] = dgc_temp.y;
  dgc[2] = dgc_temp.z;
  dgc[3] = dgc_temp.w;
  inormc[0] = inormc_temp.x;
  inormc[1] = inormc_temp.y;
  inormc[2] = inormc_temp.z;
  inormc[3] = inormc_temp.w;

  // Preload the shared memory values we will use into registers using
  // vectorized loads
  dfc_temp = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c + 1];
  dgc_temp = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c + 1];
  inormc_temp = reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c + 1];

  // Rename to array structure for ease of use
  dfc[4] = dfc_temp.x;
  dfc[5] = dfc_temp.y;
  dfc[6] = dfc_temp.z;
  dgc[4] = dgc_temp.x;
  dgc[5] = dgc_temp.y;
  dgc[6] = dgc_temp.z;
  inormc[4] = inormc_temp.x;
  inormc[5] = inormc_temp.y;
  inormc[6] = inormc_temp.z;

  // Preload the shared memory values we will use into registers using
  // vectorized loads
  VEC4_DATA_TYPE dgr_temp = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_row)[r];
  VEC4_DATA_TYPE dfr_temp = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_row)[r];
  VEC4_DATA_TYPE inormr_temp =
      reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_row)[r];

  // Rename to array structure
  dfr[0] = dfr_temp.x;
  dfr[1] = dfr_temp.y;
  dfr[2] = dfr_temp.z;
  dfr[3] = dfr_temp.w;
  dgr[0] = dgr_temp.x;
  dgr[1] = dgr_temp.y;
  dgr[2] = dgr_temp.z;
  dgr[3] = dgr_temp.w;
  inormr[0] = inormr_temp.x;
  inormr[1] = inormr_temp.y;
  inormr[2] = inormr_temp.z;
  inormr[3] = inormr_temp.w;

  // For NN profiles we need to do a vectorized load to pull the best-so-far
  // values from cache
  if (COMPUTE_ROWS) {
    switch (PROFILE_TYPE) {
      case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      case PROFILE_TYPE_MATRIX_SUMMARY:
      case PROFILE_TYPE_1NN_INDEX: {
        ulonglong4 mp_row_check_temp;
        mp_row_check_temp =
            reinterpret_cast<ulonglong4 *>(smem.local_mp_row)[r];
        mp_entry e;
        e.ulong = mp_row_check_temp.x;
        mp_row_check[0] = e.floats[0];
        e.ulong = mp_row_check_temp.y;
        mp_row_check[1] = e.floats[0];
        e.ulong = mp_row_check_temp.z;
        mp_row_check[2] = e.floats[0];
        e.ulong = mp_row_check_temp.w;
        mp_row_check[3] = e.floats[0];
        break;
      }
      case PROFILE_TYPE_1NN: {
        float4 mp_row_check_temp;
        mp_row_check_temp = reinterpret_cast<float4 *>(smem.local_mp_row)[r];
        mp_row_check[0] = mp_row_check_temp.x;
        mp_row_check[1] = mp_row_check_temp.y;
        mp_row_check[2] = mp_row_check_temp.z;
        mp_row_check[3] = mp_row_check_temp.w;
        break;
      }
      case PROFILE_TYPE_SUM_THRESH:
      default:
        break;
    }
  }

  // Generate and coalesce distances into profile
  do_row<0, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
         COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>(
      info, distc, inormc, dfc, dgc, inormr, dfr, dgr, mp_row_check, idxc, smem,
      args);
  do_row<1, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
         COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>(
      info, distc, inormc, dfc, dgc, inormr, dfr, dgr, mp_row_check, idxc, smem,
      args);
  do_row<2, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
         COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>(
      info, distc, inormc, dfc, dgc, inormr, dfr, dgr, mp_row_check, idxc, smem,
      args);
  do_row<3, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
         COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>(
      info, distc, inormc, dfc, dgc, inormr, dfr, dgr, mp_row_check, idxc, smem,
      args);

  // Update the column wise matrix profile with the best-so-far
  if (COMPUTE_COLS) {
    update_cols<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE>(
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

template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, SCAMPProfileType PROFILE_TYPE>
__device__ inline void reduce_edge_std(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem,
    SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE dist[4],
    DISTANCE_TYPE &dist_row, uint32_t &idx_row, int diag, int num_diags, int n,
    OptionalArgs &args) {
  if (info.global_col + iter < n && diag + iter < num_diags) {
    if (COMPUTE_ROWS) {
      if (dist[iter] > dist_row) {
        dist_row = dist[iter];
        idx_row = info.global_col + iter;
      }
    }
    if (COMPUTE_COLS) {
      MPatomicMax<ATOMIC_BLOCK>(
          (uint64_t *)(smem.local_mp_col + info.local_col + iter), dist[iter],
          info.global_row);
    }
  }
}

template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS>
__device__ inline void reduce_edge(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN> &smem,
    SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE dist[4],
    DISTANCE_TYPE &dist_row, uint32_t &idx_row, int diag, int num_diags, int n,
    OptionalArgs &args) {
  if (info.global_col + iter < n && diag + iter < num_diags &&
      !isnan(dist[iter])) {
    if (COMPUTE_ROWS) {
      dist_row = fmaxf(dist_row, dist[iter]);
    }
    if (COMPUTE_COLS) {
      fAtomicMax<ATOMIC_BLOCK>(
          (float *)(smem.local_mp_col + info.local_col + iter), dist[iter]);
    }
  }
}

template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS>
__device__ inline void reduce_edge(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN_INDEX> &smem,
    SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE dist[4],
    DISTANCE_TYPE &dist_row, uint32_t &idx_row, int diag, int num_diags, int n,
    OptionalArgs &args) {
  reduce_edge_std<iter, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                  COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_1NN_INDEX>(
      smem, info, dist, dist_row, idx_row, diag, num_diags, n, args);
}

template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS>
__device__ inline void reduce_edge(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_MATRIX_SUMMARY> &smem,
    SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE dist[4],
    DISTANCE_TYPE &dist_row, uint32_t &idx_row, int diag, int num_diags, int n,
    OptionalArgs &args) {
  reduce_edge_std<iter, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                  COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_MATRIX_SUMMARY>(
      smem, info, dist, dist_row, idx_row, diag, num_diags, n, args);
}

template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS>
__device__ inline void reduce_edge(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_APPROX_ALL_NEIGHBORS>
        &smem,
    SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE dist[4],
    DISTANCE_TYPE &dist_row, uint32_t &idx_row, int diag, int num_diags, int n,
    OptionalArgs &args) {
  reduce_edge_std<iter, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                  COMPUTE_ROWS, COMPUTE_COLS,
                  PROFILE_TYPE_APPROX_ALL_NEIGHBORS>(
      smem, info, dist, dist_row, idx_row, diag, num_diags, n, args);
}

template <int iter, typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename ACCUM_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS>
__device__ inline void reduce_edge(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_SUM_THRESH> &smem,
    SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE dist[4],
    DISTANCE_TYPE &dist_row, uint32_t &idx_row, int diag, int num_diags, int n,
    OptionalArgs &args) {
  if (info.global_col + iter < n && diag + iter < num_diags) {
    if (dist[iter] > args.threshold) {
      if (COMPUTE_ROWS) {
        dist_row += dist[iter];
      }
      if (COMPUTE_COLS) {
        do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
            smem.local_mp_col + info.local_col + iter, dist[iter]);
      }
    }
  }
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void reduce_row(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_SUM_THRESH> &smem,
    int row, DISTANCE_TYPE dist_row, uint32_t idx_row) {
  do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(smem.local_mp_row + row,
                                                dist_row);
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void reduce_row(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN> &smem, int row,
    DISTANCE_TYPE dist_row, uint32_t idx_row) {
  fAtomicMax<ATOMIC_BLOCK>((float *)(smem.local_mp_row + row), dist_row);
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void reduce_row(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN_INDEX> &smem,
    int row, DISTANCE_TYPE dist_row, uint32_t idx_row) {
  MPatomicMax<ATOMIC_BLOCK>((uint64_t *)(smem.local_mp_row + row), dist_row,
                            idx_row);
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void reduce_row(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_MATRIX_SUMMARY> &smem,
    int row, DISTANCE_TYPE dist_row, uint32_t idx_row) {
  MPatomicMax<ATOMIC_BLOCK>((uint64_t *)(smem.local_mp_row + row), dist_row,
                            idx_row);
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE>
__device__ inline void reduce_row(
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_APPROX_ALL_NEIGHBORS>
        &smem,
    int row, DISTANCE_TYPE dist_row, uint32_t idx_row) {
  MPatomicMax<ATOMIC_BLOCK>((uint64_t *)(smem.local_mp_row + row), dist_row,
                            idx_row);
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, SCAMPProfileType PROFILE_TYPE,
          bool COMPUTE_ROWS, bool COMPUTE_COLS>
__device__ inline void do_row_edge(
    SCAMPThreadInfo<ACCUM_TYPE> &info,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem, int n,
    int diag, int num_diags, OptionalArgs &args) {
  DISTANCE_TYPE dist_row = init_dist<DISTANCE_TYPE, PROFILE_TYPE>();
  DISTANCE_TYPE dist[4];
  int col = info.local_col;
  int row = info.local_row;
  uint32_t idx_row = 0;
  DATA_TYPE inormr = smem.inorm_row[row];
  DATA_TYPE dgr = smem.dg_row[row];
  DATA_TYPE dfr = smem.df_row[row];

  // Compute the next set of distances (row y)
  dist[0] = info.cov1 * smem.inorm_col[col] * inormr;
  dist[1] = info.cov2 * smem.inorm_col[col + 1] * inormr;
  dist[2] = info.cov3 * smem.inorm_col[col + 2] * inormr;
  dist[3] = info.cov4 * smem.inorm_col[col + 3] * inormr;

  // Update cov and compute the next distance values (row y)
  info.cov1 = info.cov1 + smem.df_col[col] * dgr + smem.dg_col[col] * dfr;
  info.cov2 =
      info.cov2 + smem.df_col[col + 1] * dgr + smem.dg_col[col + 1] * dfr;
  info.cov3 =
      info.cov3 + smem.df_col[col + 2] * dgr + smem.dg_col[col + 2] * dfr;
  info.cov4 =
      info.cov4 + smem.df_col[col + 3] * dgr + smem.dg_col[col + 3] * dfr;

  reduce_edge<0, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
              COMPUTE_ROWS, COMPUTE_COLS>(smem, info, dist, dist_row, idx_row,
                                          diag, num_diags, n, args);
  reduce_edge<1, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
              COMPUTE_ROWS, COMPUTE_COLS>(smem, info, dist, dist_row, idx_row,
                                          diag, num_diags, n, args);
  reduce_edge<2, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
              COMPUTE_ROWS, COMPUTE_COLS>(smem, info, dist, dist_row, idx_row,
                                          diag, num_diags, n, args);
  reduce_edge<3, DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
              COMPUTE_ROWS, COMPUTE_COLS>(smem, info, dist, dist_row, idx_row,
                                          diag, num_diags, n, args);

  if (COMPUTE_ROWS) {
    reduce_row<DATA_TYPE, PROFILE_DATA_TYPE, DISTANCE_TYPE>(smem, row, dist_row,
                                                            idx_row);
  }
}
