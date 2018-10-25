#include <unordered_map>
#include "kernels.h"

namespace SCAMP {

constexpr int DIAGS_PER_THREAD = 4;
constexpr int BLOCKSZ_SP = 512;
constexpr int BLOCKSZ_DP = 256;
constexpr int BLOCKSPERSM_SELF = 2;
constexpr int BLOCKSPERSM_AB = 2;
constexpr int TILE_HEIGHT_SP = 200;
constexpr int TILE_HEIGHT_DP = 200;
constexpr float CC_MIN = -FLT_MAX;

template <typename T>
struct SCAMPKernelInputArgs {
  SCAMPKernelInputArgs(const T *__restrict__ cov_, const T *__restrict__ dfa_,
                       const T *__restrict__ dfb_, const T *__restrict__ dga_,
                       const T *__restrict__ dgb_,
                       const T *__restrict__ normsa_,
                       const T *__restrict__ normsb_, uint32_t n_x_,
                       uint32_t n_y_, int32_t exclusion_lower_,
                       int32_t exclusion_upper_, OptionalArgs opt_)
      : cov(cov_),
        dfa(dfa_),
        dfb(dfb_),
        dga(dga_),
        dgb(dgb_),
        normsa(normsa_),
        normsb(normsb_),
        n_x(n_x_),
        n_y(n_y_),
        exclusion_lower(exclusion_lower_),
        exclusion_upper(exclusion_upper_),
        opt(opt_) {}
  const T *__restrict__ cov;
  const T *__restrict__ dfa;
  const T *__restrict__ dfb;
  const T *__restrict__ dga;
  const T *__restrict__ dgb;
  const T *__restrict__ normsa;
  const T *__restrict__ normsb;
  uint32_t n_x;
  uint32_t n_y;
  int32_t exclusion_lower;
  int32_t exclusion_upper;
  OptionalArgs opt;
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE>
struct SCAMPSmem {
  __device__ SCAMPSmem(char *smem, bool compute_rows, bool compute_columns,
                       int tile_width, int tile_height) {
    constexpr int data_size = sizeof(DATA_TYPE);
    constexpr int profile_size = sizeof(PROFILE_DATA_TYPE);
    int curr_byte = 0;
    df_col = (DATA_TYPE *)(smem);
    curr_byte += tile_width * data_size;
    dg_col = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_width * data_size;
    inorm_col = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_width * data_size;
    df_row = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_height * data_size;
    dg_row = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_height * data_size;
    inorm_row = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_height * data_size;
    if (compute_columns) {
      local_mp_col = (PROFILE_DATA_TYPE *)(smem + curr_byte);
      curr_byte += tile_width * profile_size;
    }
    if (compute_rows) {
      local_mp_row = (PROFILE_DATA_TYPE *)(smem + curr_byte);
      curr_byte += tile_height * profile_size;
    }
  }
  DATA_TYPE *__restrict__ df_col;
  DATA_TYPE *__restrict__ dg_col;
  DATA_TYPE *__restrict__ inorm_col;
  DATA_TYPE *__restrict__ df_row;
  DATA_TYPE *__restrict__ dg_row;
  DATA_TYPE *__restrict__ inorm_row;
  PROFILE_DATA_TYPE *__restrict__ local_mp_col;
  PROFILE_DATA_TYPE *__restrict__ local_mp_row;
};

// Atomically updates the MP/idxs using a single 64-bit integer. We lose a small
// amount of precision in the output, if we do not do this we are unable
// to atomically update both the matrix profile and the indexes without using a
// critical section and dedicated locks.
__device__ inline void MPatomicMax(volatile uint64_t *address, float val,
                                   unsigned int idx) {
  mp_entry loc, loctest;
  loc.floats[0] = val;
  loc.ints[1] = idx;
  loctest.ulong = *address;
  while (loctest.floats[0] < val) {
    loctest.ulong =
        atomicCAS((unsigned long long int *)address, loctest.ulong, loc.ulong);
  }
}

// As above, but checks a previously read value before attempting another read
// This allows us to exploit vectorized loads of the matrix profile
__device__ inline void MPatomicMax_check(
    volatile uint64_t *__restrict__ address, float val, unsigned int idx,
    float curr_val) {
  if (val > curr_val) {
    mp_entry loc, loctest;
    loc.floats[0] = val;
    loc.ints[1] = idx;
    loctest.ulong = *address;
    while (loctest.floats[0] < val) {
      loctest.ulong = atomicCAS((unsigned long long int *)address,
                                loctest.ulong, loc.ulong);
    }
  }
}

__device__ inline void MPMax(const float d1, const float d2,
                             const unsigned int i1, const unsigned int i2,
                             float &outd, unsigned int &outi) {
  if (d1 >= d2) {
    outd = d1;
    outi = i1;
  } else {
    outd = d2;
    outi = i2;
  }
}

// Computes max(a,b) with index and stores the result in a
__device__ inline void MPMax2(float &d1, const float &d2, unsigned int &i1,
                              const unsigned int &i2) {
  if (d2 > d1) {
    d1 = d2;
    i1 = i2;
  }
}

// Computes the max of 4 values in a float 4
__device__ inline float max4(const float4 &d, const unsigned int init,
                             unsigned int &idx) {
  float ret = d.x;
  idx = init;
  if (d.y > ret) {
    ret = d.y;
    idx = init + 1;
  }
  if (d.z > ret) {
    ret = d.z;
    idx = init + 2;
  }
  if (d.w > ret) {
    ret = d.w;
    idx = init + 3;
  }
  return ret;
}

class SCAMPStrategy {
 public:
};

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR INITIALIZING SHARED MEMORY
//
//
//////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class InitMemStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(SCAMPKernelInputArgs<double> &args,
                       SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                       PROFILE_DATA_TYPE *__restrict__ profile_a,
                       PROFILE_DATA_TYPE *__restrict__ profile_B,
                       uint32_t col_start, uint32_t row_start) {
    assert(false);
  }

 protected:
  __device__ InitMemStrategy() {}
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ>
class InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                      tile_width, tile_height, BLOCKSZ, PROFILE_TYPE_SUM_THRESH>
    : public SCAMPStrategy {
 public:
  __device__ InitMemStrategy() {}
  __device__ void exec(SCAMPKernelInputArgs<double> &args,
                       SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                       PROFILE_DATA_TYPE *__restrict__ profile_a,
                       PROFILE_DATA_TYPE *__restrict__ profile_B,
                       uint32_t col_start, uint32_t row_start) {
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while (local_position < tile_width && global_position < args.n_x) {
      smem.dg_col[local_position] = args.dga[global_position];
      smem.df_col[local_position] = args.dfa[global_position];
      smem.inorm_col[local_position] = args.normsa[global_position];
      if (COMPUTE_COLS) {
        smem.local_mp_col[local_position] = 0.0;
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
        smem.local_mp_row[local_position] = 0.0;
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ>
class InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                      tile_width, tile_height, BLOCKSZ, PROFILE_TYPE_1NN_INDEX>
    : public SCAMPStrategy {
 public:
  __device__ InitMemStrategy() {}
  __device__ virtual void exec(SCAMPKernelInputArgs<double> &args,
                               SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                               PROFILE_DATA_TYPE *__restrict__ profile_a,
                               PROFILE_DATA_TYPE *__restrict__ profile_B,
                               uint32_t col_start, uint32_t row_start) {
    assert(true);
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowOptStrategy : SCAMPStrategy {
 public:
  __device__ virtual void exec(
      ACCUM_TYPE &cov1, ACCUM_TYPE &cov2, ACCUM_TYPE &cov3, ACCUM_TYPE &cov4,
      DISTANCE_TYPE &distc1, DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3,
      DISTANCE_TYPE &distc4, const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, const int &row, const int &col,
      const int &global_row, const int &global_col,
      PROFILE_DATA_TYPE *__restrict__ mp_row, const OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoRowOptStrategy() {}
};

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR COMPUTING A THREAD-ROW OF THE
// DISTANCE MATRIX (common, optimized case)
//
//
//////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
class DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                       COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_SUM_THRESH>
    : public SCAMPStrategy {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ virtual inline __attribute__((always_inline)) void exec(
      ACCUM_TYPE &cov1, ACCUM_TYPE &cov2, ACCUM_TYPE &cov3, ACCUM_TYPE &cov4,
      DISTANCE_TYPE &distc1, DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3,
      DISTANCE_TYPE &distc4, const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, const int &row, const int &col,
      const int &global_row, const int &global_col,
      PROFILE_DATA_TYPE *__restrict__ mp_row, const OptionalArgs &args) {
    DISTANCE_TYPE distx = cov1 * inormcx * inormr;
    DISTANCE_TYPE disty = cov2 * inormcy * inormr;
    DISTANCE_TYPE distz = cov3 * inormcz * inormr;
    DISTANCE_TYPE distw = cov4 * inormcw * inormr;
    DISTANCE_TYPE thresh = args.threshold;

    // Compute the next covariance values
    cov1 = cov1 + df_colx * dg_row + dg_colx * df_row;
    cov2 = cov2 + df_coly * dg_row + dg_coly * df_row;
    cov3 = cov3 + df_colz * dg_row + dg_colz * df_row;
    cov4 = cov4 + df_colw * dg_row + dg_colw * df_row;

    DISTANCE_TYPE count_row = 0;

    if (distx > thresh) {
      if (COMPUTE_ROWS) {
        count_row += distx;
      }
      if (COMPUTE_COLS) {
        distc1 += distx;
      }
    }
    if (disty > thresh) {
      if (COMPUTE_ROWS) {
        count_row += disty;
      }
      if (COMPUTE_COLS) {
        distc2 += disty;
      }
    }
    if (distz > thresh) {
      if (COMPUTE_ROWS) {
        count_row += distz;
      }
      if (COMPUTE_COLS) {
        distc3 += distz;
      }
    }
    if (distw > thresh) {
      if (COMPUTE_ROWS) {
        count_row += distw;
      }
      if (COMPUTE_COLS) {
        distc4 += distw;
      }
    }
    // coalesce all row updates to lane 0 of each warp and atomically update
    // This way is more efficient than atomics when we expect a lot of updates
    if (COMPUTE_ROWS) {
#pragma unroll
      for (int i = 16; i >= 1; i /= 2) {
        count_row += __shfl_down_sync(0xffffffff, count_row, i);
      }
      if ((threadIdx.x & 0x1f) == 0) {
        atomicAdd_block(mp_row + row, count_row);
      }
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
class DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                       COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_1NN_INDEX> {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ virtual inline __attribute__((always_inline)) void exec(
      ACCUM_TYPE &cov1, ACCUM_TYPE &cov2, ACCUM_TYPE &cov3, ACCUM_TYPE &cov4,
      DISTANCE_TYPE &distc1, DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3,
      DISTANCE_TYPE &distc4, const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, const int &row, const int &col,
      const int &global_row, const int &global_col,
      PROFILE_DATA_TYPE *__restrict__ mp_row, const OptionalArgs &args) {
    assert(false);
  }
};

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR COMPUTING A THREAD-ROW OF THE
// DISTANCE MATRIX (uncommon, edge case)
//
// Slow path (edge tiles)
// Does a single iteration of the inner loop on 4 diagonals per thread, not
// unrolled Checks for the boundary case where only 1, 2, or 3 diagonals can be
// updated
//
//////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowEdgeStrategy : SCAMPStrategy {
 public:
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              const OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoRowEdgeStrategy() {}
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
// SCAMPProfileType PROFILE_TYPE, std::enable_if<PROFILE_TYPE ==
// PROFILE_TYPE_SUM_THRESH || PROFILE_TYPE ==
// PROFILE_TYPE_FREQUENCY_THRESH>::type>
class DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_SUM_THRESH>
    : SCAMPStrategy {
 public:
  __device__ DoRowEdgeStrategy() {}
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              const OptionalArgs &args) {
    DISTANCE_TYPE distr = 0;
    DISTANCE_TYPE distx, disty, distz, distw;
    DISTANCE_TYPE thresh = static_cast<DISTANCE_TYPE>(args.threshold);
    DATA_TYPE inormr = smem.inorm_row[i];
    DATA_TYPE dgr = smem.dg_row[i];
    DATA_TYPE dfr = smem.df_row[i];

    // Compute the next set of distances (row y)
    distx = cov1 * smem.inorm_col[j] * inormr;
    disty = cov2 * smem.inorm_col[j + 1] * inormr;
    distz = cov3 * smem.inorm_col[j + 2] * inormr;
    distw = cov4 * smem.inorm_col[j + 3] * inormr;
    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + smem.df_col[j] * dgr + smem.dg_col[j] * dfr;
    cov2 = cov2 + smem.df_col[j + 1] * dgr + smem.dg_col[j + 1] * dfr;
    cov3 = cov3 + smem.df_col[j + 2] * dgr + smem.dg_col[j + 2] * dfr;
    cov4 = cov4 + smem.df_col[j + 3] * dgr + smem.dg_col[j + 3] * dfr;

    if (distx > thresh) {
      if (COMPUTE_ROWS) {
        distr += distx;
      }
      if (COMPUTE_COLS) {
        atomicAdd_block(smem.local_mp_col + j, distx);
      }
    }
    if (x + 1 < n && diag + 1 < num_diags) {
      if (disty > thresh) {
        if (COMPUTE_ROWS) {
          distr += disty;
        }
        if (COMPUTE_COLS) {
          atomicAdd_block(smem.local_mp_col + j + 1, disty);
        }
      }
    }
    if (x + 2 < n && diag + 2 < num_diags) {
      if (distz > thresh) {
        if (COMPUTE_ROWS) {
          distr += distz;
        }
        if (COMPUTE_COLS) {
          atomicAdd_block(smem.local_mp_col + j + 2, distz);
        }
      }
    }
    if (x + 3 < n && diag + 3 < num_diags) {
      if (distw > thresh) {
        if (COMPUTE_ROWS) {
          distr += distw;
        }
        if (COMPUTE_COLS) {
          atomicAdd_block(smem.local_mp_col + j + 3, distw);
        }
      }
    }
    if (COMPUTE_ROWS) {
      atomicAdd_block(smem.local_mp_row + i, distr);
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
// SCAMPProfileType PROFILE_TYPE, std::enable_if<PROFILE_TYPE ==
// PROFILE_TYPE_1NN_SUM, int>::value >
class DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_1NN_INDEX>
    : SCAMPStrategy {
 public:
  __device__ DoRowEdgeStrategy() {}
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              const OptionalArgs &args) {
    assert(false);
  }
};

//////////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR UPDATING THE COLUMNS OF THE LOCAL MP VALUES IN
// THE OPTIMIZED CASE
//
//////////////////////////////////////////////////////////////////////

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class UpdateColumnsStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(DISTANCE_TYPE distc1, DISTANCE_TYPE distc2,
                       DISTANCE_TYPE distc3, DISTANCE_TYPE distc4,
                       DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
                       DISTANCE_TYPE distc7,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       uint64_t col) {
    assert(false);
  }

 protected:
  __device__ UpdateColumnsStrategy() {}
};

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE>
class UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE,
                            PROFILE_TYPE_SUM_THRESH> : public SCAMPStrategy {
 public:
  __device__ UpdateColumnsStrategy() {}
  __device__ void exec(DISTANCE_TYPE distc1, DISTANCE_TYPE distc2,
                       DISTANCE_TYPE distc3, DISTANCE_TYPE distc4,
                       DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
                       DISTANCE_TYPE distc7,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       uint64_t col) {
    int lane = threadIdx.x & 0x1f;
    DISTANCE_TYPE overlap_1, overlap_2, overlap_3;

    // Send the overlapping sums to the next thread
    overlap_1 = __shfl_up_sync(0xffffffff, distc5, 1);
    overlap_2 = __shfl_up_sync(0xffffffff, distc6, 1);
    overlap_3 = __shfl_up_sync(0xffffffff, distc7, 1);
    if (lane > 0) {
      distc1 += overlap_1;
      distc2 += overlap_2;
      distc3 += overlap_3;
    }
    // Update the shared memory sums
    atomicAdd_block(local_mp_col + col, distc1);
    atomicAdd_block(local_mp_col + col + 1, distc2);
    atomicAdd_block(local_mp_col + col + 2, distc3);
    atomicAdd_block(local_mp_col + col + 3, distc4);
    // The last thread in the warp has to make additional updates to shared
    // memory as it had nowhere to send its overlapping sums
    if (lane == 31) {
      atomicAdd_block(local_mp_col + col + 4, distc5);
      atomicAdd_block(local_mp_col + col + 5, distc6);
      atomicAdd_block(local_mp_col + col + 6, distc7);
    }
  }
};

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE>
class UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE,
                            PROFILE_TYPE_1NN_INDEX> : public SCAMPStrategy {
 public:
  __device__ UpdateColumnsStrategy() {}
  __device__ void exec(DISTANCE_TYPE distc1, DISTANCE_TYPE distc2,
                       DISTANCE_TYPE distc3, DISTANCE_TYPE distc4,
                       DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
                       DISTANCE_TYPE distc7,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       uint64_t col) {
    assert(false);
  }  // Unimplemented
};

///////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR WRITING BACK THE LOCAL MATRIX PROFILE TO MEMORY
//
///////////////////////////////////////////////////////////////////

// Dummy (forces compilation failure when the wrong types are used)
template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class WriteBackStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    assert(false);
  }

 protected:
  __device__ WriteBackStrategy() {}
};

template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
class WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS,
                        TILE_WIDTH, TILE_HEIGHT, BLOCKSZ,
                        PROFILE_TYPE_SUM_THRESH> : public SCAMPStrategy {
 public:
  __device__ WriteBackStrategy() {}
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    int global_position, local_position;
    if (COMPUTE_COLS) {
      global_position = tile_start_x + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_WIDTH && global_position < n_x) {
        atomicAdd(profile_A + global_position, local_mp_col[local_position]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
    if (COMPUTE_ROWS) {
      global_position = tile_start_y + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_HEIGHT && global_position < n_y) {
        atomicAdd(profile_B + global_position, local_mp_row[local_position]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
  }
};

template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
class WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS,
                        TILE_WIDTH, TILE_HEIGHT, BLOCKSZ,
                        PROFILE_TYPE_1NN_INDEX> : public SCAMPStrategy {
 public:
  __device__ WriteBackStrategy() {}
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    int global_position, local_position;
    if (COMPUTE_COLS) {
      global_position = tile_start_x + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_WIDTH && global_position < n_x) {
        mp_entry e;
        e.ulong = local_mp_col[local_position];
        MPatomicMax(profile_A + global_position, e.floats[0], e.ints[1]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
    if (COMPUTE_ROWS) {
      global_position = tile_start_y + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_HEIGHT && global_position < n_y) {
        mp_entry e;
        e.ulong = local_mp_row[local_position];
        MPatomicMax(profile_B + global_position, e.floats[0], e.ints[1]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////
//
//  SCAMP TACTIC DESCRIBES STRATEGY FOR WHAT OPS TO EXECUTE IN THE KERNEL
//
/////////////////////////////////////////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class SCAMPTactic {
 public:
  __device__ SCAMPTactic() {}
  __device__ void InitMem(SCAMPKernelInputArgs<double> &args,
                          SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                          PROFILE_DATA_TYPE *__restrict__ profile_a,
                          PROFILE_DATA_TYPE *__restrict__ profile_b,
                          uint32_t col_start, uint32_t row_start) {
    _init_mem.exec(args, smem, profile_a, profile_b, col_start, row_start);
  }
  __device__ inline __attribute__((always_inline)) void DoRow(
      ACCUM_TYPE &cov1, ACCUM_TYPE &cov2, ACCUM_TYPE &cov3, ACCUM_TYPE &cov4,
      DISTANCE_TYPE &distc1, DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3,
      DISTANCE_TYPE &distc4, const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, const int &row, const int &col,
      const int &global_row, const int &global_col,
      PROFILE_DATA_TYPE *__restrict__ mp_row, const OptionalArgs &args) {
    _do_row.exec(cov1, cov2, cov3, cov4, distc1, distc2, distc3, distc4,
                 inormcx, inormcy, inormcz, inormcw, inormr, df_colx, df_coly,
                 df_colz, df_colw, dg_colx, dg_coly, dg_colz, dg_colw, df_row,
                 dg_row, row, col, global_row, global_col, mp_row, args);
  }
  __device__ inline void DoEdge(int i, int j, int x, int y, int n,
                                ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                                ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                                size_t num_diags,
                                SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                                const OptionalArgs &args) {
    _do_edge.exec(i, j, x, y, n, cov1, cov2, cov3, cov4, diag, num_diags, smem,
                  args);
  }
  __device__ inline void WriteBack(uint32_t tile_start_x, uint32_t tile_start_y,
                                   uint32_t n_x, uint32_t n_y,
                                   PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                                   PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                                   PROFILE_DATA_TYPE *__restrict__ profile_A,
                                   PROFILE_DATA_TYPE *__restrict__ profile_B) {
    _do_writeback.exec(tile_start_x, tile_start_y, n_x, n_y, local_mp_col,
                       local_mp_row, profile_A, profile_B);
  }
  __device__ inline void UpdateColumns(
      DISTANCE_TYPE distc1, DISTANCE_TYPE distc2, DISTANCE_TYPE distc3,
      DISTANCE_TYPE distc4, DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
      DISTANCE_TYPE distc7, PROFILE_DATA_TYPE *__restrict__ local_mp_col,
      uint64_t col) {
    _update_cols.exec(distc1, distc2, distc3, distc4, distc5, distc6, distc7,
                      local_mp_col, col);
  }

 private:
  InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                  TILE_WIDTH, TILE_HEIGHT, BLOCKSZ, PROFILE_TYPE>
      _init_mem;
  DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                   COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>
      _do_row;
  UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE>
      _update_cols;
  DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                    COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>
      _do_edge;
  WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS, TILE_WIDTH,
                    TILE_HEIGHT, BLOCKSZ, PROFILE_TYPE>
      _do_writeback;
};

///////////////////////////////////////////////////////////////////////////////
// OPTIMIZED CODE PATH:
// do_unrolled_row4 is the optimized matrix profile code path which computes
// one row of work for a single thread. It is specialized for each profile type
// that is computed.
// do_iteration_unroll_4 computes a 4x4 block of the distance matrix by calling
// do_unrolled_row4 four separate times.
// We are computing a tile that looks like this:
// C:1 2 3 4 5 6 7
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
          typename SCAMP_TACTIC>
__device__ inline void do_iteration_unroll_4(
    int i, int j, int x, int y, ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
    ACCUM_TYPE &cov3, ACCUM_TYPE &cov4,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem, const OptionalArgs &args,
    const DISTANCE_TYPE dist_initializer, SCAMP_TACTIC &tactic) {
  DISTANCE_TYPE distc1 = dist_initializer;
  DISTANCE_TYPE distc2 = dist_initializer;
  DISTANCE_TYPE distc3 = dist_initializer;
  DISTANCE_TYPE distc4 = dist_initializer;
  DISTANCE_TYPE distc5 = dist_initializer;
  DISTANCE_TYPE distc6 = dist_initializer;
  DISTANCE_TYPE distc7 = dist_initializer;

  // Load row values 2 at a time, load column values 4 at a time
  int r = i >> 1;
  int c = j >> 2;

  // Preload the shared memory values we will use into registers
  // We load 4 values per thread into a double4 vector type
  VEC4_DATA_TYPE dfc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c];
  VEC4_DATA_TYPE dgc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c];
  VEC4_DATA_TYPE inormc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c];
  VEC4_DATA_TYPE dfc2 = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c + 1];
  VEC4_DATA_TYPE dgc2 = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c + 1];
  VEC4_DATA_TYPE inormc2 =
      reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c + 1];

  // Due to a lack of registers on volta, we only load these row values 2 at a
  // time
  VEC2_DATA_TYPE dgr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.dg_row)[r];
  VEC2_DATA_TYPE dfr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.df_row)[r];
  VEC2_DATA_TYPE inormr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.inorm_row)[r];

  // Do rows one at a time:
  tactic.DoRow(cov1, cov2, cov3, cov4, distc1, distc2, distc3, distc4, inormc.x,
               inormc.y, inormc.z, inormc.w, inormr.x, dfc.x, dfc.y, dfc.z,
               dfc.w, dgc.x, dgc.y, dgc.z, dgc.w, dfr.x, dgr.x, i, j, y, x,
               smem.local_mp_row, args);

  tactic.DoRow(cov1, cov2, cov3, cov4, distc2, distc3, distc4, distc5, inormc.y,
               inormc.z, inormc.w, inormc2.x, inormr.y, dfc.y, dfc.z, dfc.w,
               dfc2.x, dgc.y, dgc.z, dgc.w, dgc2.x, dfr.y, dgr.y, i + 1, j + 1,
               y + 1, x + 1, smem.local_mp_row, args);

  // Load the values for the next 2 rows
  dgr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.dg_row)[r + 1];
  dfr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.df_row)[r + 1];
  inormr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.inorm_row)[r + 1];

  tactic.DoRow(cov1, cov2, cov3, cov4, distc3, distc4, distc5, distc6, inormc.z,
               inormc.w, inormc2.x, inormc2.y, inormr.x, dfc.z, dfc.w, dfc2.x,
               dfc2.y, dgc.z, dgc.w, dgc2.x, dgc2.y, dfr.x, dgr.x, i + 2, j + 2,
               y + 2, x + 2, smem.local_mp_row, args);

  tactic.DoRow(cov1, cov2, cov3, cov4, distc4, distc5, distc6, distc7, inormc.w,
               inormc2.x, inormc2.y, inormc2.z, inormr.y, dfc.w, dfc2.x, dfc2.y,
               dfc2.z, dgc.w, dgc2.x, dgc2.y, dgc2.z, dfr.y, dgr.y, i + 3,
               j + 3, y + 3, x + 3, smem.local_mp_row, args);

  if (COMPUTE_COLS) {
    tactic.UpdateColumns(distc1, distc2, distc3, distc4, distc5, distc6, distc7,
                         smem.local_mp_col, j);
  }
}

///////////////////////////////////////
// Slow path (edge tiles)
// Does a single iteration of the inner loop on 4 diagonals per thread, not
// unrolled Checks for the boundary case where only 1, 2, or 3 diagonals can be
// updated
//////////////////////////////////////
template <typename DATA_TYPE, typename VEC4_DATA_TYPE, typename ACCUM_TYPE,
          typename PROFILE_DATA_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS>
__device__ inline void do_iteration_edge_sum(
    int i, int j, int x, int y, int n, ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
    ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag, size_t num_diags,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem, const OptionalArgs &args) {
  assert(false);
}

template <
    typename DATA_TYPE, typename VEC4_DATA_TYPE, typename ACCUM_TYPE,
    typename PROFILE_DATA_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
    bool COMPUTE_COLS,
    std::enable_if_t<true == (std::is_integral<PROFILE_DATA_TYPE>::value &&
                              std::is_integral<DISTANCE_TYPE>::value) ||
                     (std::is_floating_point<PROFILE_DATA_TYPE>::value &&
                      std::is_floating_point<DISTANCE_TYPE>::value)>>
__device__ inline void do_iteration_edge_sum(
    int i, int j, int x, int y, int n, ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
    ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag, size_t num_diags,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem, const OptionalArgs &args) {
  DISTANCE_TYPE distr = 0;
  DISTANCE_TYPE distx, disty, distz, distw;
  DISTANCE_TYPE thresh = static_cast<DISTANCE_TYPE>(args.threshold);
  DATA_TYPE inormr = smem.inorm_row[i];
  DATA_TYPE dgr = smem.dg_row[i];
  DATA_TYPE dfr = smem.df_row[i];

  // Compute the next set of distances (row y)
  distx = cov1 * smem.inorm_col[j] * inormr;
  disty = cov2 * smem.inorm_col[j + 1] * inormr;
  distz = cov3 * smem.inorm_col[j + 2] * inormr;
  distw = cov4 * smem.inorm_col[j + 3] * inormr;
  // Update cov and compute the next distance values (row y)
  cov1 = cov1 + smem.df_col[j] * dgr + smem.dg_col[j] * dfr;
  cov2 = cov2 + smem.df_col[j + 1] * dgr + smem.dg_col[j + 1] * dfr;
  cov3 = cov3 + smem.df_col[j + 2] * dgr + smem.dg_col[j + 2] * dfr;
  cov4 = cov4 + smem.df_col[j + 3] * dgr + smem.dg_col[j + 3] * dfr;

  if (distx > thresh) {
    if (COMPUTE_ROWS) {
      distr += distx;
    }
    if (COMPUTE_COLS) {
      atomicAdd_block(smem.local_mp_col + j, distx);
    }
  }
  if (x + 1 < n && diag + 1 < num_diags) {
    if (disty > thresh) {
      if (COMPUTE_ROWS) {
        distr += disty;
      }
      if (COMPUTE_COLS) {
        atomicAdd_block(smem.local_mp_col + j + 1, disty);
      }
    }
  }
  if (x + 2 < n && diag + 2 < num_diags) {
    if (distz > thresh) {
      if (COMPUTE_ROWS) {
        distr += distz;
      }
      if (COMPUTE_COLS) {
        atomicAdd_block(smem.local_mp_col + j + 2, distz);
      }
    }
  }
  if (x + 3 < n && diag + 3 < num_diags) {
    if (distw > thresh) {
      if (COMPUTE_ROWS) {
        distr += distw;
      }
      if (COMPUTE_COLS) {
        atomicAdd_block(smem.local_mp_col + j + 3, distw);
      }
    }
  }
  if (COMPUTE_ROWS) {
    atomicAdd_block(smem.local_mp_row + i, distr);
  }
}

template <typename DATA_TYPE, typename VEC4_DATA_TYPE, typename ACCUM_TYPE,
          typename PROFILE_DATA_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, SCAMPProfileType PROFILE_TYPE>
__device__ inline void do_iteration_edge_switch(
    int i, int j, int x, int y, int n, ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
    ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag, size_t num_diags,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem, const OptionalArgs &args) {
  switch (PROFILE_TYPE) {
    case PROFILE_TYPE_SUM_THRESH:
    case PROFILE_TYPE_FREQUENCY_THRESH:
      do_iteration_edge_sum<DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                            PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                            COMPUTE_COLS>(i, j, x, y, n, cov1, cov2, cov3, cov4,
                                          diag, num_diags, smem, args);
      return;
    case PROFILE_TYPE_1NN_INDEX:
      /*
              do_iteration_edge_1NN<DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
         COMPUTE_ROWS, COMPUTE_COLS>( i, j, x, y, global_start_x,
         global_start_y, n, cov1, cov2, cov3, cov4, df_col, df_row, dg_col,
         dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row, diag,
         num_diags, args); return;
      */
    default:
      assert(false);
      return;
  }
}

template <class T, class VT4, class ACC, bool mixed, bool full_join,
          bool only_col>
__device__ inline void do_iteration_4diag(
    int i, int j, int x, int y, size_t global_start_x, size_t global_start_y,
    int n, ACC &cov1, ACC &cov2, ACC &cov3, ACC &cov4, T *__restrict__ df_col,
    T *__restrict__ df_row, T *__restrict__ dg_col, T *__restrict__ dg_row,
    T *__restrict__ inorm_col, T *__restrict__ inorm_row,
    double *__restrict__ local_mp_col, double *__restrict__ local_mp_row,
    size_t diag, size_t num_diags, double thresh) {
  double count_rolling = 0;
  double4 dist;

  T inormr = inorm_row[i];
  T dgr = dg_row[i];
  T dfr = df_row[i];

  // Compute the next set of distances (row y)
  dist.x = cov1 * inorm_col[j] * inormr;
  dist.y = cov2 * inorm_col[j + 1] * inormr;
  dist.z = cov3 * inorm_col[j + 2] * inormr;
  dist.w = cov4 * inorm_col[j + 3] * inormr;
  // Update cov and compute the next distance values (row y)
  cov1 = cov1 + df_col[j] * dgr + dg_col[j] * dfr;
  cov2 = cov2 + df_col[j + 1] * dgr + dg_col[j + 1] * dfr;
  cov3 = cov3 + df_col[j + 2] * dgr + dg_col[j + 2] * dfr;
  cov4 = cov4 + df_col[j + 3] * dgr + dg_col[j + 3] * dfr;

  if (dist.x > thresh) {
    count_rolling += dist.x;
    if (full_join || only_col) {
      atomicAdd_block(local_mp_col + j, dist.x);
    }
  }
  if (x + 1 < n && diag + 1 < num_diags) {
    if (dist.y > thresh) {
      count_rolling += dist.y;
      if (full_join || only_col) {
        atomicAdd_block(local_mp_col + j + 1, dist.y);
      }
    }
  }
  if (x + 2 < n && diag + 2 < num_diags) {
    if (dist.z > thresh) {
      count_rolling += dist.z;
      if (full_join || only_col) {
        atomicAdd_block(local_mp_col + j + 2, dist.z);
      }
    }
  }
  if (x + 3 < n && diag + 3 < num_diags) {
    if (dist.w > thresh) {
      count_rolling += dist.w;
      if (full_join || only_col) {
        atomicAdd_block(local_mp_col + j + 3, dist.w);
      }
    }
  }
  if (full_join || !only_col) {
    atomicAdd_block(local_mp_row + i, count_rolling);
  }
}

template <typename DISTANCE_TYPE, SCAMPProfileType profile_type>
__device__ constexpr DISTANCE_TYPE getInitializer() {
  // static_assert(is_valid_profile_type(profile_type), "Invalid profile type
  // specified in kernel");
  switch (profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
    case PROFILE_TYPE_FREQUENCY_THRESH:
      return static_cast<DISTANCE_TYPE>(0);
    case PROFILE_TYPE_1NN_INDEX:
      return static_cast<DISTANCE_TYPE>(CC_MIN);
    default:
      assert(false);
      // Should never happen
      return 0;
  }
}

// Computes the matrix profile given the sliding dot products for the first
// query and the precomputed data statisics
template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE, int blocks_per_sm, int tile_height,
          int BLOCKSZ>
__global__ void __launch_bounds__(BLOCKSZ, blocks_per_sm)
    do_tile(SCAMPKernelInputArgs<double> args,
            PROFILE_DATA_TYPE *__restrict__ profile_A,
            PROFILE_DATA_TYPE *__restrict__ profile_B) {
  constexpr int diags_per_thread = 4;
  constexpr int tile_width = tile_height + BLOCKSZ * diags_per_thread;
  SCAMPTactic<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
              COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height, BLOCKSZ,
              PROFILE_TYPE>
      tactic;
  constexpr DISTANCE_TYPE initializer =
      getInitializer<DISTANCE_TYPE, PROFILE_TYPE>();

  extern __shared__ char smem_raw[];
  SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> smem(
      smem_raw, COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height);

  const unsigned int start_diag = (threadIdx.x * diags_per_thread) +
                                  blockIdx.x * (blockDim.x * diags_per_thread);

  // This is the index of the meta-diagonal that this thread block will work on
  const unsigned int meta_diagonal_idx = blockIdx.x;

  // The first threads are acutally computing the trivial match between the same
  // subsequence we exclude these from the calculation
  int tile_start_x =
      meta_diagonal_idx * (BLOCKSZ * diags_per_thread) + args.exclusion_lower;
  int tile_start_y = 0;

  // x is the global column of the distance matrix
  // y is the global row of the distance matrix
  // localX, localY are the local coordinates of the thread position in the tile
  // it is working on
  int x = tile_start_x + threadIdx.x * diags_per_thread;
  int y = 0;

  // Each thread updates 2 diagonals at once
  ACCUM_TYPE cov1, cov2, cov3, cov4;

  const unsigned int num_diags = args.n_x - args.exclusion_upper;

  // Load the first dot product values
  if (x < args.n_x) {
    cov1 = args.cov[x];
  }

  if (x + 1 < args.n_x) {
    cov2 = args.cov[x + 1];
  }

  if (x + 2 < args.n_x) {
    cov3 = args.cov[x + 2];
  }

  if (x + 3 < args.n_x) {
    cov4 = args.cov[x + 3];
  }

  /////////////////////////////////////
  // Main loop
  /////////////////////////////////////
  // Each threadblock finds all the distances on a 'metadiagonal'
  // We use a tiled approach for each thread block
  // The tiles are horizontal slices of the diagonal, think of a parallelogram
  // cut from a diagonal slice of the distance matrix Each thread starts on the
  // first row and works its way down-right towards right side of the distance
  // matrix
  while (tile_start_x < args.n_x && tile_start_y < args.n_y) {
    // Initialize the next tile's shared memory
    tactic.InitMem(args, smem, profile_A, profile_B, tile_start_x,
                   tile_start_y);
    // initialize_tile_memory_switch<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS,
    // COMPUTE_COLS, tile_width, tile_height, BLOCKSZ, PROFILE_TYPE>(args, smem,
    // profile_A, profile_B, tile_start_x, tile_start_y);
    // Start of new tile, sync
    __syncthreads();

    // There are 2 pathways here, most of the time we take the fast path (top),
    // the last block will take the slower path as well as the fast path
    // (bottom)
    if (tile_start_x + tile_width < args.n_x &&
        tile_start_y + tile_height < args.n_y &&
        start_diag + diags_per_thread - 1 < num_diags) {
      for (int i = 0, j = threadIdx.x * diags_per_thread; i < tile_height;
           i += diags_per_thread, j += diags_per_thread) {
        do_iteration_unroll_4<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE,
                              ACCUM_TYPE, PROFILE_DATA_TYPE, DISTANCE_TYPE,
                              COMPUTE_ROWS, COMPUTE_COLS, decltype(tactic)>(
            i, j, x + i, y + i, cov1, cov2, cov3, cov4, smem, args.opt,
            initializer, tactic);
      }
      x += tile_height;
      y += tile_height;
    } else if (start_diag < num_diags) {
      int localX = threadIdx.x * diags_per_thread;
      int localY = 0;
      while (x < args.n_x && y < args.n_y && localY < tile_height) {
        tactic.DoEdge(localY, localX, x, y, args.n_x, cov1, cov2, cov3, cov4,
                      start_diag, num_diags, smem, args.opt);

        ++x;
        ++y;
        ++localX;
        ++localY;
      }
    }

    // After this sync, the caches will be updated with the best so far values
    // for this tile
    __syncthreads();

    tactic.WriteBack(tile_start_x, tile_start_y, args.n_x, args.n_y,
                     smem.local_mp_col, smem.local_mp_row, profile_A,
                     profile_B);

    // Update the tile position
    tile_start_x += tile_height;
    tile_start_y += tile_height;

    // Make sure our updates were committed before we pull in the next tile
    __threadfence_block();
  }
}

int get_diags_per_thread(bool fp64, const cudaDeviceProp &dev_prop) {
  return 4;
}

int get_blocksz(SCAMPPrecisionType t, const cudaDeviceProp &dev_prop) {
  if (t == PRECISION_DOUBLE) {
    return BLOCKSZ_DP;
  } else {
    return BLOCKSZ_SP;
  }
}

SCAMPError_t kernel_ab_join_upper(
    const double *QT, const double *timeseries_A, const double *timeseries_B,
    const double *df_A, const double *df_B, const double *dg_A,
    const double *dg_B, const double *norms_A, const double *norms_B,
    double *profile_A, double *profile_B, size_t window_size, size_t tile_width,
    size_t tile_height, size_t global_x, size_t global_y, size_t global_start_x,
    size_t global_start_y, const cudaDeviceProp &props, SCAMPPrecisionType t,
    bool full_join, double thresh, cudaStream_t s) {
  /*
    int diags_per_thread = get_diags_per_thread(t, props);
    int blocksz = get_blocksz(t, props);
    dim3 grid(1, 1, 1);
    dim3 block(blocksz, 1, 1);
    int num_workers = ceil(tile_width / (float)diags_per_thread);
    grid.x = ceil(num_workers / (double)blocksz);
    int smem;
    if (full_join) {
      // We can have an exclusion zone if this ab join is part of a larger
      // self-join
      int exclusion = window_size / 4;
      if (global_y + global_start_y >= global_x + global_start_x &&
          global_start_y + global_y <= global_start_x + global_x + exclusion) {
        num_workers = ceil((tile_width - exclusion) / (float)diags_per_thread);
        grid.x = ceil(num_workers / (double)blocksz);
      } else {
        exclusion = 0;
      }
      if (tile_width <= exclusion) {
        return SCAMP_NO_ERROR;
      }
      switch (t) {
        case PRECISION_DOUBLE:
          smem = get_smem<double>(TILE_HEIGHT_DP, t, true, true, props);
          do_tile<double, double2, double4, double, false, true, true,
                  BLOCKSPERSM_AB, TILE_HEIGHT_DP, BLOCKSZ_DP>
              <<<grid, block, smem, s>>>(
                  QT, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A,
                  profile_B, window_size, tile_width, tile_height, global_x,
                  global_y, exclusion, 0, thresh);
          break;
        case PRECISION_MIXED:
          smem = get_smem<float>(TILE_HEIGHT, t, true, true, props);
          do_tile<float, float2, float4, double, true, true, true,
    BLOCKSPERSM_AB, TILE_HEIGHT, BLOCKSZ_SP><<<grid, block, smem, s>>>( QT,
    df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A, profile_B, window_size,
    tile_width, tile_height, global_x, global_y, exclusion, 0, thresh); break;
        case PRECISION_SINGLE:
          smem = get_smem<float>(TILE_HEIGHT, t, true, true, props);
          do_tile<float, float2, float4, float, false, true, true,
    BLOCKSPERSM_AB, TILE_HEIGHT, BLOCKSZ_SP><<<grid, block, smem, s>>>( QT,
    df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A, profile_B, window_size,
    tile_width, tile_height, global_x, global_y, exclusion, 0, thresh); break;
        default:
          break;
      }
    } else {
      switch (t) {
        case PRECISION_DOUBLE:
          smem = get_smem<double>(TILE_HEIGHT_DP, t, false, true, props);
          do_tile<double, double2, double4, double, false, false, true,
                  BLOCKSPERSM_AB, TILE_HEIGHT_DP, BLOCKSZ_DP>
              <<<grid, block, smem, s>>>(QT, df_A, df_B, dg_A, dg_B, norms_A,
                                         norms_B, profile_A, profile_B,
                                         window_size, tile_width, tile_height,
                                         global_x, global_y, 0, 0, thresh);
          break;
        case PRECISION_MIXED:
          smem = get_smem<float>(TILE_HEIGHT, t, false, true, props);
          do_tile<float, float2, float4, double, true, false, true,
                  BLOCKSPERSM_AB, TILE_HEIGHT, BLOCKSZ_SP>
              <<<grid, block, smem, s>>>(QT, df_A, df_B, dg_A, dg_B, norms_A,
                                         norms_B, profile_A, profile_B,
                                         window_size, tile_width, tile_height,
                                         global_x, global_y, 0, 0, thresh);
          break;
        case PRECISION_SINGLE:
          smem = get_smem<float>(TILE_HEIGHT, t, false, true, props);
          do_tile<float, float2, float4, float, false, false, true,
                  BLOCKSPERSM_AB, TILE_HEIGHT, BLOCKSZ_SP>
              <<<grid, block, smem, s>>>(QT, df_A, df_B, dg_A, dg_B, norms_A,
                                         norms_B, profile_A, profile_B,
                                         window_size, tile_width, tile_height,
                                         global_x, global_y, 0, 0, thresh);
          break;
        default:
          break;
      }
    }
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
      return SCAMP_CUDA_ERROR;
    }
  */
  return SCAMP_NO_ERROR;
}

SCAMPError_t kernel_ab_join_lower(
    const double *QT, const double *timeseries_A, const double *timeseries_B,
    const double *df_A, const double *df_B, const double *dg_A,
    const double *dg_B, const double *norms_A, const double *norms_B,
    double *profile_A, double *profile_B, size_t window_size, size_t tile_width,
    size_t tile_height, size_t global_x, size_t global_y, size_t global_start_x,
    size_t global_start_y, const cudaDeviceProp &props, SCAMPPrecisionType t,
    bool full_join, double thresh, cudaStream_t s) {
  /*
    int diags_per_thread = get_diags_per_thread(t, props);
    int blocksz = get_blocksz(t, props);
    dim3 grid(1, 1, 1);
    dim3 block(blocksz, 1, 1);
    int num_workers = ceil(tile_height / (float)diags_per_thread);
    grid.x = ceil(num_workers / (double)blocksz);
    int smem;
    if (full_join) {
      // We can have an exclusion zone if this ab join is part of a larger
      // self-join
      int exclusion = window_size / 4;
      if (global_y + global_start_y + tile_height >= global_x + global_start_x
    && global_y + global_start_y + tile_height <= global_x + global_start_x +
    exclusion) { num_workers = ceil((tile_height - exclusion) /
    (float)diags_per_thread); grid.x = ceil(num_workers / (double)blocksz); }
    else { exclusion = 0;
      }
      if (tile_height <= exclusion) {
        return SCAMP_NO_ERROR;
      }
      switch (t) {
        case PRECISION_DOUBLE:
          smem = get_smem<double>(TILE_HEIGHT_DP, t, true, true, props);
          do_tile<double, double2, double4, double, false, true, true,
                  BLOCKSPERSM_AB, TILE_HEIGHT_DP, BLOCKSZ_DP>
              <<<grid, block, smem, s>>>(
                  QT, df_B, df_A, dg_B, dg_A, norms_B, norms_A, profile_B,
                  profile_A, window_size, tile_height, tile_width, global_y,
                  global_x, 0, exclusion, thresh);
          break;
        case PRECISION_MIXED:
          smem = get_smem<float>(TILE_HEIGHT, t, true, true, props);
          do_tile<float, float2, float4, double, true, true, true,
    BLOCKSPERSM_AB, TILE_HEIGHT, BLOCKSZ_SP><<<grid, block, smem, s>>>( QT,
    df_B, df_A, dg_B, dg_A, norms_B, norms_A, profile_B, profile_A, window_size,
    tile_height, tile_width, global_y, global_x, 0, exclusion, thresh); break;
        case PRECISION_SINGLE:
          smem = get_smem<float>(TILE_HEIGHT, t, true, true, props);
          do_tile<float, float2, float4, float, false, true, true,
    BLOCKSPERSM_AB, TILE_HEIGHT, BLOCKSZ_SP><<<grid, block, smem, s>>>( QT,
    df_B, df_A, dg_B, dg_A, norms_B, norms_A, profile_B, profile_A, window_size,
    tile_height, tile_width, global_y, global_x, 0, exclusion, thresh); break;
        default:
          return SCAMP_CUDA_ERROR;
      }
    } else {
      switch (t) {
        case PRECISION_DOUBLE:
          smem = get_smem<double>(TILE_HEIGHT_DP, t, false, false, props);
          do_tile<double, double2, double4, double, false, false, false,
                  BLOCKSPERSM_AB, TILE_HEIGHT_DP, BLOCKSZ_DP>
              <<<grid, block, smem, s>>>(QT, df_B, df_A, dg_B, dg_A, norms_B,
                                         norms_A, profile_B, profile_A,
                                         window_size, tile_height, tile_width,
                                         global_y, global_x, 0, 0, thresh);
          break;
        case PRECISION_MIXED:
          smem = get_smem<float>(TILE_HEIGHT, t, false, false, props);
          do_tile<float, float2, float4, double, true, false, false,
                  BLOCKSPERSM_AB, TILE_HEIGHT, BLOCKSZ_SP>
              <<<grid, block, smem, s>>>(QT, df_B, df_A, dg_B, dg_A, norms_B,
                                         norms_A, profile_B, profile_A,
                                         window_size, tile_height, tile_width,
                                         global_y, global_x, 0, 0, thresh);
          break;
        case PRECISION_SINGLE:
          smem = get_smem<float>(TILE_HEIGHT, t, false, false, props);
          do_tile<float, float2, float4, float, false, false, false,
                  BLOCKSPERSM_AB, TILE_HEIGHT, BLOCKSZ_SP>
              <<<grid, block, smem, s>>>(QT, df_B, df_A, dg_B, dg_A, norms_B,
                                         norms_A, profile_B, profile_A,
                                         window_size, tile_height, tile_width,
                                         global_y, global_x, 0, 0, thresh);
          break;
        default:
          return SCAMP_CUDA_ERROR;
      }
    }
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
      return SCAMP_CUDA_ERROR;
    }
  */
  return SCAMP_NO_ERROR;
}

int get_exclusion(uint64_t window_size, uint64_t start_row,
                  uint64_t start_column) {
  int exclusion = window_size / 4;
  if (!(start_column >= start_row && start_column <= start_row + exclusion)) {
    return 0;
  }
  return window_size / 4;
}

constexpr int FPTypeSize(SCAMPPrecisionType dtype) {
  static_assert(sizeof(float) == 4, "Float is assumed to be 4 bytes");
  static_assert(sizeof(double) == 8, "Double is assumed to be 8 bytes");
  switch (dtype) {
    case PRECISION_DOUBLE:
      return 8;
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      return 4;
    case PRECISION_INVALID:
      return -1;
  }
  return -1;
}

int GetTileHeight(SCAMPPrecisionType dtype) {
  switch (dtype) {
    case PRECISION_DOUBLE:
      return TILE_HEIGHT_DP;
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      return TILE_HEIGHT_SP;
    case PRECISION_INVALID:
      return -1;
  }
  return -1;
}

template <class PROFILE_DATA_TYPE>
int get_smem(bool computing_rows, bool computing_cols, int blocksz,
             SCAMPPrecisionType intermediate_data_type) {
  constexpr int diags_per_thread = 4;
  constexpr int num_shared_variables = 3;
  int intermediate_data_size = FPTypeSize(intermediate_data_type);
  printf("%d\n", intermediate_data_type);
  int tile_height = GetTileHeight(intermediate_data_type);
  int tile_width = blocksz * diags_per_thread + tile_height;
  printf("%d, height = %d, width = %d\n", intermediate_data_size, tile_height,
         tile_width);
  int smem = (tile_width + tile_height) * num_shared_variables *
             intermediate_data_size;
  if (computing_cols) {
    smem += tile_width * sizeof(PROFILE_DATA_TYPE);
  }
  if (computing_rows) {
    smem += tile_height * sizeof(PROFILE_DATA_TYPE);
  }
  return smem;
}

template <typename PROFILE_DATA_TYPE, SCAMPProfileType PROFILE_TYPE,
          int BLOCKSPERSM>
SCAMPError_t LaunchDoTile(SCAMPKernelInputArgs<double> args,
                          PROFILE_DATA_TYPE *profile_A,
                          PROFILE_DATA_TYPE *profile_B,
                          SCAMPPrecisionType fp_type, bool computing_rows,
                          uint64_t blocksz, uint64_t num_blocks, uint64_t smem,
                          cudaStream_t s) {
  dim3 block(blocksz, 1, 1);
  dim3 grid(num_blocks, 1, 1);
  constexpr bool COMPUTE_COLS = true;
  if (computing_rows) {
    constexpr bool COMPUTE_ROWS = true;
    switch (fp_type) {
      case PRECISION_DOUBLE: {
        do_tile<double, double2, double4, double, PROFILE_DATA_TYPE, double,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, float,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, float,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }

      default:
        return SCAMP_CUDA_ERROR;
    }
    return SCAMP_NO_ERROR;
  }
  constexpr bool COMPUTE_ROWS = false;
  switch (fp_type) {
    case PRECISION_DOUBLE: {
      do_tile<double, double2, double4, double, PROFILE_DATA_TYPE, double,
              COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
              TILE_HEIGHT_DP, BLOCKSZ_DP>
          <<<grid, block, smem, s>>>(args, profile_A, profile_B);
      break;
    }
    case PRECISION_MIXED: {
      do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, float,
              COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
              TILE_HEIGHT_SP, BLOCKSZ_SP>
          <<<grid, block, smem, s>>>(args, profile_A, profile_B);
      break;
    }
    case PRECISION_SINGLE: {
      do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, float,
              COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
              TILE_HEIGHT_SP, BLOCKSZ_SP>
          <<<grid, block, smem, s>>>(args, profile_A, profile_B);
      break;
    }
    default:
      return SCAMP_CUDA_ERROR;
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t kernel_self_join_upper(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_x, uint64_t global_y,
    const cudaDeviceProp &props, SCAMPPrecisionType t, const OptionalArgs &args,
    SCAMPProfileType profile_type, cudaStream_t s) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  int32_t exclusion = get_exclusion(window_size, global_x, global_y);
  uint64_t num_workers =
      ceil((tile_width - exclusion) / (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(QT, df_A, df_B, dg_A, dg_B, norms_A,
                                         norms_B, tile_width, tile_height,
                                         exclusion, (int32_t)0, args);
  uint64_t smem;
  if (exclusion < tile_width) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        smem = get_smem<double>(true, true, blocksz, t);
        printf("blocksz = %d, SMEM upper = %d\n", blocksz, smem);
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        smem = get_smem<uint64_t>(true, true, blocksz, t);
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<uint64_t *>(
                profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<uint64_t *>(
                profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, blocksz, num_blocks, smem, s);
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t kernel_self_join_lower(
    const double *QT, const double *df_A, const double *df_B,
    const double *dg_A, const double *dg_B, const double *norms_A,
    const double *norms_B, DeviceProfile *profile_A, DeviceProfile *profile_B,
    size_t window_size, size_t tile_width, size_t tile_height, size_t global_x,
    size_t global_y, const cudaDeviceProp &props, SCAMPPrecisionType t,
    const OptionalArgs &args, SCAMPProfileType profile_type, cudaStream_t s) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  uint64_t exclusion =
      get_exclusion(window_size, global_x, global_y + tile_height);
  uint64_t num_workers =
      ceil((tile_height - exclusion) / (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  uint64_t smem;
  SCAMPKernelInputArgs<double> tile_args(QT, df_B, df_A, dg_B, dg_A, norms_B,
                                         norms_A, tile_height, tile_width, 0,
                                         exclusion, args);
  if (exclusion < tile_height) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        smem = get_smem<double>(true, true, blocksz, t);
        printf("blocksz = %d, SMEM lower = %d\n", blocksz, smem);
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        smem = get_smem<uint64_t>(true, true, blocksz, t);
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<uint64_t *>(
                profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<uint64_t *>(
                profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, blocksz, num_blocks, smem, s);
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
