#pragma once

#include "kernels_inner_loop.h"

///////////////////////////////////////////////////////////////////////////////
// OPTIMIZED CODE PATH:
// do_unrolled_row4 is the optimized matrix profile code path which computes
// one row of work for a single thread. It is specialized for each profile type
// that is computed.
// do_iteration_unroll_4 computes a 4x4 block of the distance matrix by calling
// do_unrolled_row4 four separate times.
// We are computing a tile that looks like this:
// C: 1 2 3 4 5 6 7
// R1 X X X X
// R2   X X X X
// R3     X X X X
// R4       X X X X
// For 4 diagonals unrolled 4 times we compute a total of 16 distances.
// These distances cover 4 possible rows and 7 possible columns.
///////////////////////////////////////////////////////////////////////////////
// Processes 4 iterations of the inner loop. Each thread computes 4 distances
// per iteration (x,y), (x+1,y), (x+2,y), and (x+3,y) This function assumes that
// the edge cases that occur on the edge of the distance matrix are not present.
// This is the faster path, with less conditional branching.
template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE, typename = void>
class DoIterationStrategy : public SCAMPStrategy {
 public:
  __device__ inline void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoIterationStrategy() {}
};

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoIterationStrategy<
    DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE, PROFILE_DATA_TYPE,
    DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE,
    std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH>>
    : public SCAMPStrategy {
 public:
  __device__ DoIterationStrategy() {}
  __device__ void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                       SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                       OptionalArgs &args) {
    DISTANCE_TYPE distc1 = 0;
    DISTANCE_TYPE distc2 = 0;
    DISTANCE_TYPE distc3 = 0;
    DISTANCE_TYPE distc4 = 0;
    DISTANCE_TYPE distc5 = 0;
    DISTANCE_TYPE distc6 = 0;
    DISTANCE_TYPE distc7 = 0;

    // Load row values 2 at a time, load column values 4 at a time
    int r = info.local_row >> 1;
    int c = info.local_col >> 2;

    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a double4 vector type
    VEC4_DATA_TYPE dfc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c];
    VEC4_DATA_TYPE dgc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c];
    VEC4_DATA_TYPE inormc =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c];
    VEC4_DATA_TYPE dfc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c + 1];
    VEC4_DATA_TYPE dgc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c + 1];
    VEC4_DATA_TYPE inormc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c + 1];

    // Due to a lack of registers, we only load these row values 2 at a
    // time
    VEC2_DATA_TYPE dgr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.dg_row)[r];
    VEC2_DATA_TYPE dfr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.df_row)[r];
    VEC2_DATA_TYPE inormr =
        reinterpret_cast<VEC2_DATA_TYPE *>(smem.inorm_row)[r];

    // Do rows one at a time:
    _do_row.exec(info, distc1, distc2, distc3, distc4, inormc.x, inormc.y,
                 inormc.z, inormc.w, inormr.x, dfc.x, dfc.y, dfc.z, dfc.w,
                 dgc.x, dgc.y, dgc.z, dgc.w, dfr.x, dgr.x, smem, args);

    _do_row.exec(info, distc2, distc3, distc4, distc5, inormc.y, inormc.z,
                 inormc.w, inormc2.x, inormr.y, dfc.y, dfc.z, dfc.w, dfc2.x,
                 dgc.y, dgc.z, dgc.w, dgc2.x, dfr.y, dgr.y, smem, args);

    // Load the values for the next 2 rows
    dgr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.dg_row)[r + 1];
    dfr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.df_row)[r + 1];
    inormr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.inorm_row)[r + 1];

    _do_row.exec(info, distc3, distc4, distc5, distc6, inormc.z, inormc.w,
                 inormc2.x, inormc2.y, inormr.x, dfc.z, dfc.w, dfc2.x, dfc2.y,
                 dgc.z, dgc.w, dgc2.x, dgc2.y, dfr.x, dgr.x, smem, args);

    _do_row.exec(info, distc4, distc5, distc6, distc7, inormc.w, inormc2.x,
                 inormc2.y, inormc2.z, inormr.y, dfc.w, dfc2.x, dfc2.y, dfc2.z,
                 dgc.w, dgc2.x, dgc2.y, dgc2.z, dfr.y, dgr.y, smem, args);

    if (COMPUTE_COLS) {
      _update_cols.exec(distc1, distc2, distc3, distc4, distc5, distc6, distc7,
                        smem.local_mp_col, info.local_col - DIAGS_PER_THREAD);
    }
    info.global_col += DIAGS_PER_THREAD;
    info.global_row += DIAGS_PER_THREAD;
  }

 private:
  DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                   COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>
      _do_row;
  UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE>
      _update_cols;
};

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoIterationStrategy<
    DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE, PROFILE_DATA_TYPE,
    DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE,
    std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX>>
    : public SCAMPStrategy {
 public:
  __device__ DoIterationStrategy() {}
  __device__ void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                       SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                       OptionalArgs &args) {
    float4 distc = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);
    float4 distc2 = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);
    uint4 idxc, idxc2;

    // Load row values 2 at a time, load column values 4 at a time
    int r = info.local_row >> 2;
    int c = info.local_col >> 2;

    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a float4 vector type
    VEC4_DATA_TYPE dfc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c];
    VEC4_DATA_TYPE dgc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c];
    VEC4_DATA_TYPE inormc =
        (reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c]);
    VEC4_DATA_TYPE dfc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c + 1];
    VEC4_DATA_TYPE dgc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c + 1];
    VEC4_DATA_TYPE inormc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c + 1];
    ulonglong4 mp_row_check;

    // Copy the pieces of the cache we will use into registers with vectorized
    // loads
    if (COMPUTE_ROWS) {
      mp_row_check = reinterpret_cast<ulonglong4 *>(smem.local_mp_row)[r];
    }

    // Due to a lack of registers on volta, we only load these row values 2 at a
    // time
    VEC4_DATA_TYPE dgr = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_row)[r];
    VEC4_DATA_TYPE dfr = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_row)[r];
    VEC4_DATA_TYPE inormr =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_row)[r];

    mp_entry e;
    e.ulong = mp_row_check.x;
    // Do rows one at a time:
    _do_row.exec(info, distc.x, distc.y, distc.z, distc.w, idxc.x, idxc.y,
                 idxc.z, idxc.w, inormc.x, inormc.y, inormc.z, inormc.w,
                 inormr.x, dfc.x, dfc.y, dfc.z, dfc.w, dgc.x, dgc.y, dgc.z,
                 dgc.w, dfr.x, dgr.x, e.floats[0], smem, args);

    e.ulong = mp_row_check.y;
    _do_row.exec(info, distc.y, distc.z, distc.w, distc2.x, idxc.y, idxc.z,
                 idxc.w, idxc2.x, inormc.y, inormc.z, inormc.w, inormc2.x,
                 inormr.y, dfc.y, dfc.z, dfc.w, dfc2.x, dgc.y, dgc.z, dgc.w,
                 dgc2.x, dfr.y, dgr.y, e.floats[0], smem, args);

    e.ulong = mp_row_check.z;
    _do_row.exec(info, distc.z, distc.w, distc2.x, distc2.y, idxc.z, idxc.w,
                 idxc2.x, idxc2.y, inormc.z, inormc.w, inormc2.x, inormc2.y,
                 inormr.z, dfc.z, dfc.w, dfc2.x, dfc2.y, dgc.z, dgc.w, dgc2.x,
                 dgc2.y, dfr.z, dgr.z, e.floats[0], smem, args);

    e.ulong = mp_row_check.w;
    _do_row.exec(info, distc.w, distc2.x, distc2.y, distc2.z, idxc.w, idxc2.x,
                 idxc2.y, idxc2.z, inormc.w, inormc2.x, inormc2.y, inormc2.z,
                 inormr.w, dfc.w, dfc2.x, dfc2.y, dfc2.z, dgc.w, dgc2.x, dgc2.y,
                 dgc2.z, dfr.w, dgr.w, e.floats[0], smem, args);

    // After the 4th row, we have completed columns 4, 5, 6, and 7
    if (COMPUTE_COLS) {
      ulonglong4 mp_col_check1, mp_col_check2;
      mp_col_check1 = reinterpret_cast<ulonglong4 *>(smem.local_mp_col)[c];
      mp_col_check2 = reinterpret_cast<ulonglong4 *>(smem.local_mp_col)[c + 1];
      e.ulong = mp_col_check1.x;
      MPatomicMax_check(smem.local_mp_col + info.local_col - 4, distc.x, idxc.x,
                        e.floats[0]);
      e.ulong = mp_col_check1.y;
      MPatomicMax_check(smem.local_mp_col + info.local_col - 3, distc.y, idxc.y,
                        e.floats[0]);
      e.ulong = mp_col_check1.z;
      MPatomicMax_check(smem.local_mp_col + info.local_col - 2, distc.z, idxc.z,
                        e.floats[0]);
      e.ulong = mp_col_check1.w;
      MPatomicMax_check(smem.local_mp_col + info.local_col - 1, distc.w, idxc.w,
                        e.floats[0]);
      e.ulong = mp_col_check2.x;
      MPatomicMax_check(smem.local_mp_col + info.local_col, distc2.x, idxc2.x,
                        e.floats[0]);
      e.ulong = mp_col_check2.y;
      MPatomicMax_check(smem.local_mp_col + info.local_col + 1, distc2.y,
                        idxc2.y, e.floats[0]);
      e.ulong = mp_col_check2.z;
      MPatomicMax_check(smem.local_mp_col + info.local_col + 2, distc2.z,
                        idxc2.z, e.floats[0]);
    }
  }

 private:
  DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, float,
                   COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>
      _do_row;
};

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoIterationStrategy<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                          PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                          COMPUTE_COLS, PROFILE_TYPE,
                          std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_1NN>>
    : public SCAMPStrategy {
 public:
  __device__ DoIterationStrategy() {}
  __device__ void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                       SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                       OptionalArgs &args) {
    float4 distc = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);
    float4 distc2 = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);

    // Load row values 2 at a time, load column values 4 at a time
    int r = info.local_row >> 2;
    int c = info.local_col >> 2;

    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a float4 vector type
    VEC4_DATA_TYPE dfc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c];
    VEC4_DATA_TYPE dgc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c];
    VEC4_DATA_TYPE inormc =
        (reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c]);
    VEC4_DATA_TYPE dfc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c + 1];
    VEC4_DATA_TYPE dgc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c + 1];
    VEC4_DATA_TYPE inormc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c + 1];
    float4 mp_row_check;

    // Copy the pieces of the cache we will use into registers with vectorized
    // loads
    if (COMPUTE_ROWS) {
      mp_row_check = reinterpret_cast<float4 *>(smem.local_mp_row)[r];
    }

    // Due to a lack of registers on volta, we only load these row values 2 at a
    // time
    VEC4_DATA_TYPE dgr = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_row)[r];
    VEC4_DATA_TYPE dfr = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_row)[r];
    VEC4_DATA_TYPE inormr =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_row)[r];

    // Do rows one at a time:
    _do_row.exec(info, distc.x, distc.y, distc.z, distc.w, inormc.x, inormc.y,
                 inormc.z, inormc.w, inormr.x, dfc.x, dfc.y, dfc.z, dfc.w,
                 dgc.x, dgc.y, dgc.z, dgc.w, dfr.x, dgr.x, mp_row_check.x, smem,
                 args);

    _do_row.exec(info, distc.y, distc.z, distc.w, distc2.x, inormc.y, inormc.z,
                 inormc.w, inormc2.x, inormr.y, dfc.y, dfc.z, dfc.w, dfc2.x,
                 dgc.y, dgc.z, dgc.w, dgc2.x, dfr.y, dgr.y, mp_row_check.y,
                 smem, args);

    _do_row.exec(info, distc.z, distc.w, distc2.x, distc2.y, inormc.z, inormc.w,
                 inormc2.x, inormc2.y, inormr.z, dfc.z, dfc.w, dfc2.x, dfc2.y,
                 dgc.z, dgc.w, dgc2.x, dgc2.y, dfr.z, dgr.z, mp_row_check.z,
                 smem, args);

    _do_row.exec(info, distc.w, distc2.x, distc2.y, distc2.z, inormc.w,
                 inormc2.x, inormc2.y, inormc2.z, inormr.w, dfc.w, dfc2.x,
                 dfc2.y, dfc2.z, dgc.w, dgc2.x, dgc2.y, dgc2.z, dfr.w, dgr.w,
                 mp_row_check.w, smem, args);

    // After the 4th row, we have completed columns 4, 5, 6, and 7
    if (COMPUTE_COLS) {
      float4 mp_col_check1, mp_col_check2;
      mp_col_check1 = reinterpret_cast<float4 *>(smem.local_mp_col)[c];
      mp_col_check2 = reinterpret_cast<float4 *>(smem.local_mp_col)[c + 1];
      fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col + info.local_col - 4,
                                     distc.x, mp_col_check1.x);
      fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col + info.local_col - 3,
                                     distc.y, mp_col_check1.y);
      fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col + info.local_col - 2,
                                     distc.z, mp_col_check1.z);
      fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col + info.local_col - 1,
                                     distc.w, mp_col_check1.w);
      fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col + info.local_col,
                                     distc2.x, mp_col_check2.x);
      fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col + info.local_col + 1,
                                     distc2.y, mp_col_check2.y);
      fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_col + info.local_col + 2,
                                     distc2.z, mp_col_check2.z);
    }
  }

 private:
  DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, float,
                   COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>
      _do_row;
};
