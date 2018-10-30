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

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE, typename PROFILE_DATA_TYPE, typename VEC2_PROFILE_TYPE, typename VEC4_PROFILE_TYPE>
struct SCAMPSmem {
  __device__ SCAMPSmem(char* smem, bool compute_rows, bool compute_cols, int tile_width, int tile_height) : inorm_row_offset(0), df_row_offset(tile_height), dg_row_offset(tile_height * 2)
  {
    int total_vals = 0;
    if (sizeof(DATA_TYPE) == 8) {
        _input_data_cols_a = (DATA_TYPE*) smem;
        _input_data_cols_b = _input_data_cols_a + 3 * (tile_width >> 1);
        _input_data_cols = nullptr;
        _input_data_rows = _input_data_cols_a  + 3 * tile_width;
    } else {
        _input_data_cols_a = nullptr;
        _input_data_cols_b = nullptr;
        _input_data_cols = (DATA_TYPE*) smem;
        _input_data_rows = _input_data_cols + 3 * tile_width;
    }
    
    if (compute_cols && compute_rows) {
        if (sizeof(PROFILE_DATA_TYPE) == 8) {
            _mp_col_a = (PROFILE_DATA_TYPE*)(_input_data_rows + 3 * tile_height);
            _mp_col_b = _mp_col_a + (tile_width >> 1);
            _mp_col = nullptr;
            _mp_row = _mp_col_a + tile_width;
        } else {
            _mp_col = (PROFILE_DATA_TYPE*)(_input_data_rows + 3 * tile_height);
            _mp_row = _mp_col + tile_width;
        }
   
    } else if (compute_cols) {
        if (sizeof(PROFILE_DATA_TYPE) == 8) {
            _mp_col_a = (PROFILE_DATA_TYPE*)(_input_data_rows + 3 * tile_height);
            _mp_col_b = _mp_col_a + (tile_width >> 1);
            _mp_col = nullptr;
        } else {
            _mp_col = (PROFILE_DATA_TYPE*)(_input_data_rows + 3 * tile_height);
        }
        _mp_row = nullptr;
    } else if (compute_rows) {
        _mp_col = nullptr; 
        _mp_col_a = nullptr; 
        _mp_col_b = nullptr; 
        _mp_row = (PROFILE_DATA_TYPE*)(_input_data_rows + 3 * tile_height);
    } 
  } 
  __device__ inline VEC4_DATA_TYPE get_col_vec4(int chunk, int offset) {
        if (sizeof(DATA_TYPE) == 8) {
          VEC2_DATA_TYPE partial1 = reinterpret_cast<VEC2_DATA_TYPE*>(_input_data_cols_a + (chunk * stride + offset))[0];
          VEC2_DATA_TYPE partial2 = reinterpret_cast<VEC2_DATA_TYPE*>(_input_data_cols_b + (chunk * stride + offset))[0];
          return {partial1.x, partial1.y, partial2.x, partial2.y};  
        } 
        return reinterpret_cast<VEC4_DATA_TYPE*>(_input_data_cols + (2 * (chunk * stride + offset)))[0];
  }
  __device__ inline DATA_TYPE get_col(int full_position, int offset) {
        int chunk = full_position >> 2;
        int pos = full_position & 3;
        if (sizeof(DATA_TYPE) == 8) {
          int partition = pos >> 1;
          int part_pos = pos & 1;
          if (partition == 0) {
            return _input_data_cols_a[chunk * stride + offset + part_pos];
          }
          return _input_data_cols_b[chunk * stride + offset + part_pos];
        }
        return _input_data_cols[2* (chunk * stride + offset) + pos];
  }
  __device__ inline void set_col(int full_position, DATA_TYPE value, int offset) {
        int chunk = full_position >> 2;
        int pos = full_position & 3;
        if (sizeof(DATA_TYPE) == 8) {
            int partition = pos >> 1;
            int part_pos = pos & 1;
            if (partition == 0) {
                _input_data_cols_a[chunk * stride + offset + part_pos] = value;
              
            } else {
                _input_data_cols_b[chunk * stride + offset + part_pos] = value;
            }
            return;
        }
        _input_data_cols[2 * (chunk * stride + offset) + pos] = value;
  }
  __device__ inline DATA_TYPE get_row(int full_position, int offset) {
        return _input_data_rows[offset + full_position];
  }
  __device__ inline VEC4_DATA_TYPE get_row_vec4(int chunk, int offset) {
        return reinterpret_cast<VEC4_DATA_TYPE*>(_input_data_rows + offset)[chunk];
  }
  __device__ inline void set_row(int full_position, DATA_TYPE value, int offset) {
        _input_data_rows[offset + full_position] = value;
  }
  __device__ inline void set_mp_col(int full_position, PROFILE_DATA_TYPE value) {
        int chunk = full_position >> 2;
        int pos = full_position & 3;
        if (sizeof(PROFILE_DATA_TYPE) == 8) {
            int partition = pos >> 1;
            int part_pos = pos & 1;
            if (partition == 0) {
                _mp_col_a[chunk * 2 +  part_pos] = value;
              
            } else {
                _mp_col_b[chunk * 2 + part_pos] = value;
            }
            return;
        }
        _mp_col[2 * (chunk * 2) + pos] = value; 
  }
  __device__ inline PROFILE_DATA_TYPE get_mp_col(int full_position) {
        int chunk = full_position >> 2;
        int pos = full_position & 3;
        if (sizeof(PROFILE_DATA_TYPE) == 8) {
          int partition = pos >> 1;
          int part_pos = pos & 1;
          if (partition == 0) {
            return _mp_col_a[chunk * 2 + part_pos];
          }
          return _mp_col_b[chunk * 2 + part_pos];
        }
        return _mp_col[2* (chunk * 2) + pos];
  }
  __device__ inline PROFILE_DATA_TYPE* get_mp_col_addr(int full_position) {
        int chunk = full_position >> 2;
        int pos = full_position & 3;
        if (sizeof(PROFILE_DATA_TYPE) == 8) {
          int partition = pos >> 1;
          int part_pos = pos & 1;
          if (partition == 0) {
            return _mp_col_a + chunk * 2 + part_pos;
          }
          return _mp_col_b + chunk * 2 + part_pos;
        }
        return _mp_col + 2 * (chunk * 2) + pos;
  }
  __device__ inline VEC4_PROFILE_TYPE get_mp_col_vec4(int chunk) {
        if (sizeof(PROFILE_DATA_TYPE) == 8) {
          VEC2_PROFILE_TYPE partial1 = reinterpret_cast<VEC2_PROFILE_TYPE*>(_mp_col_a + (chunk * 2))[0];
          VEC2_PROFILE_TYPE partial2 = reinterpret_cast<VEC2_PROFILE_TYPE*>(_mp_col_b + (chunk * 2))[0];
          return {partial1.x, partial1.y, partial2.x, partial2.y};  
        } 
        return reinterpret_cast<VEC4_PROFILE_TYPE*>(_mp_col + (2 * (chunk * 2)))[0];
  }
  __device__ inline PROFILE_DATA_TYPE get_mp_row(int full_position) {
        return _mp_row[full_position];
  }
  __device__ inline VEC4_PROFILE_TYPE get_mp_row_vec4(int chunk) {
        return reinterpret_cast<VEC4_PROFILE_TYPE*>(_mp_row)[chunk];
  }
  __device__ inline void set_mp_row(int full_position, PROFILE_DATA_TYPE value) {
       _mp_row[full_position] = value;
  }
  __device__ inline PROFILE_DATA_TYPE* get_mp_row_addr(int full_position) {
        return _mp_row + full_position;
  }
  const int stride = 6;
  const int inorm_col_offset = 0;
  const int df_col_offset =  2;
  const int dg_col_offset =  4;

  const int inorm_row_offset;
  const int df_row_offset;
  const int dg_row_offset;
  DATA_TYPE *__restrict__ _input_data_cols_a;
  DATA_TYPE *__restrict__ _input_data_cols_b;
  DATA_TYPE *__restrict__ _input_data_rows;
  DATA_TYPE *__restrict__ _input_data_cols;
  PROFILE_DATA_TYPE *__restrict__ _mp_col_a;
  PROFILE_DATA_TYPE *__restrict__ _mp_col_b;
  PROFILE_DATA_TYPE *__restrict__ _mp_col;
  PROFILE_DATA_TYPE *__restrict__ _mp_row;

};

template <typename ACCUM_TYPE>
struct SCAMPThreadInfo {
  ACCUM_TYPE cov1;
  ACCUM_TYPE cov2;
  ACCUM_TYPE cov3;
  ACCUM_TYPE cov4;
  uint32_t local_row;
  uint32_t local_col;
  uint32_t global_row;
  uint32_t global_col;
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
template <typename T>
__device__ inline void MPMax2(T &d1, const T &d2, unsigned int &i1,
                              const unsigned int &i2) {
  if (d2 > d1) {
    d1 = d2;
    i1 = i2;
  }
}
template <typename T>
__device__ inline T max4(T &d1, T &d2, T &d3, T &d4, const uint32_t init,
                         uint32_t &idx) {
  float ret = d1;
  idx = init;
  if (d2 > ret) {
    ret = d2;
    idx = init + 1;
  }
  if (d3 > ret) {
    ret = d3;
    idx = init + 2;
  }
  if (d4 > ret) {
    ret = d4;
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
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ, typename SMEM_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class InitMemStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(SCAMPKernelInputArgs<double> &args,
                       SMEM_TYPE &smem,
                       PROFILE_DATA_TYPE *__restrict__ profile_a,
                       PROFILE_DATA_TYPE *__restrict__ profile_B,
                       uint32_t col_start, uint32_t row_start) {
    assert(false);
  }

 protected:
  __device__ InitMemStrategy() {}
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ, typename SMEM_TYPE>
class InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                      tile_width, tile_height, BLOCKSZ, SMEM_TYPE, PROFILE_TYPE_SUM_THRESH>
    : public SCAMPStrategy {
 public:
  __device__ InitMemStrategy() {}
  __device__ void exec(SCAMPKernelInputArgs<double> &args,
                       SMEM_TYPE &smem,
                       PROFILE_DATA_TYPE *__restrict__ profile_a,
                       PROFILE_DATA_TYPE *__restrict__ profile_B,
                       uint32_t col_start, uint32_t row_start) {
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while (local_position < tile_width && global_position < args.n_x) {
      smem.set_col(local_position, args.dga[global_position], smem.dg_col_offset);
      smem.set_col(local_position, args.dfa[global_position], smem.df_col_offset);
      smem.set_col(local_position, args.normsa[global_position], smem.inorm_col_offset);
      if (COMPUTE_COLS) {
        smem.set_mp_col(local_position, 0.0);
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }

    global_position = row_start + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < tile_height && global_position < args.n_y) {
      smem.set_row(local_position, args.dgb[global_position], smem.dg_row_offset);
      smem.set_row(local_position, args.dfb[global_position], smem.df_row_offset);
      smem.set_row(local_position, args.normsb[global_position], smem.inorm_row_offset);
      if (COMPUTE_ROWS) {
        smem.set_mp_row(local_position, 0.0);
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ, typename SMEM_TYPE>
class InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                      tile_width, tile_height, BLOCKSZ, SMEM_TYPE, PROFILE_TYPE_1NN_INDEX>
    : public SCAMPStrategy {
 public:
  __device__ InitMemStrategy() {}
  __device__ virtual void exec(SCAMPKernelInputArgs<double> &args,
                               SMEM_TYPE &smem,
                               PROFILE_DATA_TYPE *__restrict__ profile_a,
                               PROFILE_DATA_TYPE *__restrict__ profile_b,
                               uint32_t col_start, uint32_t row_start) {
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while (local_position < tile_width && global_position < args.n_x) {
      smem.set_col(local_position, args.dga[global_position], smem.dg_col_offset);
      smem.set_col(local_position, args.dfa[global_position], smem.df_col_offset);
      smem.set_col(local_position, args.normsa[global_position], smem.inorm_col_offset);
      if (COMPUTE_COLS) {
        smem.set_mp_col(local_position, profile_a[global_position]);
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }

    global_position = row_start + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < tile_height && global_position < args.n_y) {
      smem.set_row(local_position, args.dgb[global_position], smem.dg_row_offset);
      smem.set_row(local_position, args.dfb[global_position], smem.df_row_offset);
      smem.set_row(local_position, args.normsb[global_position], smem.inorm_row_offset);
      if (COMPUTE_ROWS) {
        smem.set_mp_row(local_position, profile_b[global_position]);
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class DoRowOptStrategy : SCAMPStrategy {
 public:
  __device__ void exec(
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
      SMEM_TYPE &smem, OptionalArgs &args) {
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
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE>
class DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                       COMPUTE_ROWS, COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE_SUM_THRESH>
    : public SCAMPStrategy {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE &distc1,
      DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3, DISTANCE_TYPE &distc4,
      const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, SMEM_TYPE &smem,

      OptionalArgs &args) {
    DISTANCE_TYPE distx = info.cov1 * inormcx * inormr;
    DISTANCE_TYPE disty = info.cov2 * inormcy * inormr;
    DISTANCE_TYPE distz = info.cov3 * inormcz * inormr;
    DISTANCE_TYPE distw = info.cov4 * inormcw * inormr;
    DISTANCE_TYPE thresh = args.threshold;

    // Compute the next covariance values
    info.cov1 = info.cov1 + df_colx * dg_row + dg_colx * df_row;
    info.cov2 = info.cov2 + df_coly * dg_row + dg_coly * df_row;
    info.cov3 = info.cov3 + df_colz * dg_row + dg_colz * df_row;
    info.cov4 = info.cov4 + df_colw * dg_row + dg_colw * df_row;

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
        atomicAdd_block(smem.get_mp_row_addr(info.local_row), count_row);
      }
    }
    info.local_row++;
    info.local_col++;
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE>
class DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                       COMPUTE_ROWS, COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE_1NN_INDEX> {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE &distc1,
      DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3, DISTANCE_TYPE &distc4,
      uint32_t &idxc1, uint32_t &idxc2, uint32_t &idxc3, uint32_t &idxc4,
      const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, float &curr_mp_row_val,
      SMEM_TYPE &smem, OptionalArgs &args) {
    DISTANCE_TYPE distx = static_cast<DATA_TYPE>(info.cov1) * inormcx * inormr;
    DISTANCE_TYPE disty = static_cast<DATA_TYPE>(info.cov2) * inormcy * inormr;
    DISTANCE_TYPE distz = static_cast<DATA_TYPE>(info.cov3) * inormcz * inormr;
    DISTANCE_TYPE distw = static_cast<DATA_TYPE>(info.cov4) * inormcw * inormr;

    // Compute the next covariance values
    info.cov1 = info.cov1 + df_colx * dg_row + dg_colx * df_row;
    info.cov2 = info.cov2 + df_coly * dg_row + dg_coly * df_row;
    info.cov3 = info.cov3 + df_colz * dg_row + dg_colz * df_row;
    info.cov4 = info.cov4 + df_colw * dg_row + dg_colw * df_row;
    // Update the column best-so-far values

    if (COMPUTE_COLS) {
      MPMax2<DISTANCE_TYPE>(distc1, distx, idxc1, info.global_row);
      MPMax2<DISTANCE_TYPE>(distc2, disty, idxc2, info.global_row);
      MPMax2<DISTANCE_TYPE>(distc3, distz, idxc3, info.global_row);
      MPMax2<DISTANCE_TYPE>(distc4, distw, idxc4, info.global_row);
    }

    if (COMPUTE_ROWS) {
      // We take the maximum of the columns we computed for the row
      // And use that value to check the matrix profile
      uint32_t idx;
      DISTANCE_TYPE d =
          max4<DISTANCE_TYPE>(distx, disty, distz, distw, info.global_col, idx);
      MPatomicMax_check(smem.get_mp_row_addr(info.local_row), d,
                        idx, curr_mp_row_val);
    }

    info.local_row++;
    info.local_col++;
    info.global_row++;
    info.global_col++;
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
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class DoRowEdgeStrategy : SCAMPStrategy {
 public:
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SMEM_TYPE &mem,
                              OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoRowEdgeStrategy() {}
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE>
// SCAMPProfileType PROFILE_TYPE, std::enable_if<PROFILE_TYPE ==
// PROFILE_TYPE_SUM_THRESH || PROFILE_TYPE ==
// PROFILE_TYPE_FREQUENCY_THRESH>::type>
class DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                        COMPUTE_ROWS, COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE_SUM_THRESH>
    : SCAMPStrategy {
 public:
  __device__ DoRowEdgeStrategy() {}
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SMEM_TYPE &smem,
                              OptionalArgs &args) {
    DISTANCE_TYPE distr = 0;
    DISTANCE_TYPE distx, disty, distz, distw;
    DISTANCE_TYPE thresh = static_cast<DISTANCE_TYPE>(args.threshold);
    DATA_TYPE inormr = smem.get_row(i, smem.inorm_row_offset);
    DATA_TYPE dgr = smem.get_row(i, smem.dg_row_offset);
    DATA_TYPE dfr = smem.get_row(i, smem.df_row_offset);

    // Compute the next set of distances (row y)
    distx = cov1 * smem.get_col(j, smem.inorm_col_offset) * inormr;
    disty = cov2 * smem.get_col(j + 1, smem.inorm_col_offset) * inormr;
    distz = cov3 * smem.get_col(j + 2, smem.inorm_col_offset) * inormr;
    distw = cov4 * smem.get_col(j + 3, smem.inorm_col_offset) * inormr;
    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + smem.get_col(j, smem.df_col_offset) * dgr + smem.get_col(j, smem.dg_col_offset) * dfr;
    cov2 = cov2 + smem.get_col(j + 1, smem.df_col_offset) * dgr + smem.get_col(j + 1, smem.dg_col_offset) * dfr;
    cov3 = cov3 + smem.get_col(j + 2, smem.df_col_offset) * dgr + smem.get_col(j + 2, smem.dg_col_offset) * dfr;
    cov4 = cov4 + smem.get_col(j + 3, smem.df_col_offset) * dgr + smem.get_col(j + 3, smem.dg_col_offset) * dfr;

    if (distx > thresh) {
      if (COMPUTE_ROWS) {
        distr += distx;
      }
      if (COMPUTE_COLS) {
        atomicAdd_block(smem.get_mp_col_addr(j), distx);
      }
    }
    if (x + 1 < n && diag + 1 < num_diags) {
      if (disty > thresh) {
        if (COMPUTE_ROWS) {
          distr += disty;
        }
        if (COMPUTE_COLS) {
          atomicAdd_block(smem.get_mp_col_addr(j + 1), disty);
        }
      }
    }
    if (x + 2 < n && diag + 2 < num_diags) {
      if (distz > thresh) {
        if (COMPUTE_ROWS) {
          distr += distz;
        }
        if (COMPUTE_COLS) {
          atomicAdd_block(smem.get_mp_col_addr(j + 2), distz);
        }
      }
    }
    if (x + 3 < n && diag + 3 < num_diags) {
      if (distw > thresh) {
        if (COMPUTE_ROWS) {
          distr += distw;
        }
        if (COMPUTE_COLS) {
          atomicAdd_block(smem.get_mp_col_addr(j + 3), distw);
        }
      }
    }
    if (COMPUTE_ROWS) {
      atomicAdd_block(smem.get_mp_row_addr(i), distr);
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE>
// SCAMPProfileType PROFILE_TYPE, std::enable_if<PROFILE_TYPE ==
// PROFILE_TYPE_1NN_SUM, int>::value >
class DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                        COMPUTE_ROWS, COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE_1NN_INDEX>
    : SCAMPStrategy {
 public:
  __device__ DoRowEdgeStrategy() {}
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SMEM_TYPE &smem,
                              OptionalArgs &args) {
    float dist_row;
    uint32_t idx_row;
    float distx;
    float disty;
    float distz;
    float distw;

    DATA_TYPE inormr = smem.get_row(i, smem.inorm_row_offset );
    DATA_TYPE dgr = smem.get_row(i, smem.dg_row_offset);
    DATA_TYPE dfr = smem.get_row(i, smem.df_row_offset);

    // Compute the next set of distances (row y)
    distx = cov1 * smem.get_col(j, smem.inorm_col_offset) * inormr;
    disty = cov2 * smem.get_col(j + 1, smem.inorm_col_offset) * inormr;
    distz = cov3 * smem.get_col(j + 2, smem.inorm_col_offset) * inormr;
    distw = cov4 * smem.get_col(j + 3, smem.inorm_col_offset) * inormr;

    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + smem.get_col(j, smem.df_col_offset) * dgr + smem.get_col(j, smem.dg_col_offset) * dfr;
    cov2 = cov2 + smem.get_col(j + 1, smem.df_col_offset) * dgr + smem.get_col(j + 1, smem.dg_col_offset) * dfr;
    cov3 = cov3 + smem.get_col(j + 2, smem.df_col_offset) * dgr + smem.get_col(j + 2, smem.dg_col_offset) * dfr;
    cov4 = cov4 + smem.get_col(j + 3, smem.df_col_offset) * dgr + smem.get_col(j + 3, smem.dg_col_offset) * dfr;

    if (COMPUTE_COLS) {
      MPatomicMax(smem.get_mp_col_addr(j), distx, y);
    }
    dist_row = distx;
    idx_row = x;
    if (x + 1 < n && diag + 1 < num_diags) {
      if (COMPUTE_ROWS) {
        MPMax(dist_row, disty, idx_row, x + 1, dist_row, idx_row);
      }
      if (COMPUTE_COLS) {
        MPatomicMax(smem.get_mp_col_addr(j + 1), disty, y);
      }
    }
    if (x + 2 < n && diag + 2 < num_diags) {
      if (COMPUTE_ROWS) {
        MPMax(dist_row, distz, idx_row, x + 2, dist_row, idx_row);
      }
      if (COMPUTE_COLS) {
        MPatomicMax(smem.get_mp_col_addr(j + 2), distz, y);
      }
    }
    if (x + 3 < n && diag + 3 < num_diags) {
      if (COMPUTE_ROWS) {
        MPMax(dist_row, distw, idx_row, x + 3, dist_row, idx_row);
      }
      if (COMPUTE_COLS) {
        MPatomicMax(smem.get_mp_col_addr(j + 3), distw, y);
      }
    }
    if (COMPUTE_ROWS) {
      MPatomicMax(smem.get_mp_row_addr(i), dist_row, idx_row);
    }
  }
};

//////////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR UPDATING THE COLUMNS OF THE LOCAL MP VALUES IN
// THE OPTIMIZED CASE
//
//////////////////////////////////////////////////////////////////////

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE, typename SMEM_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class UpdateColumnsStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(DISTANCE_TYPE distc1, DISTANCE_TYPE distc2,
                       DISTANCE_TYPE distc3, DISTANCE_TYPE distc4,
                       DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
                       DISTANCE_TYPE distc7,
                       SMEM_TYPE &smem,
                       uint64_t col) {
    assert(false);
  }

 protected:
  __device__ UpdateColumnsStrategy() {}
};

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE, typename SMEM_TYPE>
class UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE, SMEM_TYPE,
                            PROFILE_TYPE_SUM_THRESH> : public SCAMPStrategy {
 public:
  __device__ UpdateColumnsStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      DISTANCE_TYPE distc1, DISTANCE_TYPE distc2, DISTANCE_TYPE distc3,
      DISTANCE_TYPE distc4, DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
      DISTANCE_TYPE distc7, SMEM_TYPE &smem,
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
    atomicAdd_block(smem.get_mp_col_addr(col), distc1);
    atomicAdd_block(smem.get_mp_col_addr(col + 1), distc2);
    atomicAdd_block(smem.get_mp_col_addr(col + 2), distc3);
    atomicAdd_block(smem.get_mp_col_addr(col + 3), distc4);
    // The last thread in the warp has to make additional updates to shared
    // memory as it had nowhere to send its overlapping sums
    if (lane == 31) {
      atomicAdd_block(smem.get_mp_col_addr(col + 4), distc5);
      atomicAdd_block(smem.get_mp_col_addr(col + 5), distc6);
      atomicAdd_block(smem.get_mp_col_addr(col + 6), distc7);
    }
  }
};

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE, typename SMEM_TYPE>
class UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE, SMEM_TYPE,
                            PROFILE_TYPE_1NN_INDEX> : public SCAMPStrategy {
 public:
  __device__ UpdateColumnsStrategy() {}
  __device__ void exec(DISTANCE_TYPE distc1, DISTANCE_TYPE distc2,
                       DISTANCE_TYPE distc3, DISTANCE_TYPE distc4,
                       DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
                       DISTANCE_TYPE distc7,
                       SMEM_TYPE &smem,
                       uint64_t col) {
    assert(false);
  }
};

///////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR WRITING BACK THE LOCAL MATRIX PROFILE TO MEMORY
//
///////////////////////////////////////////////////////////////////

// Dummy (forces compilation failure when the wrong types are used)
template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ, typename SMEM_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class WriteBackStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       SMEM_TYPE &smem,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    assert(false);
  }

 protected:
  __device__ WriteBackStrategy() {}
};

template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ, typename SMEM_TYPE>
class WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS,
                        TILE_WIDTH, TILE_HEIGHT, BLOCKSZ, SMEM_TYPE,
                        PROFILE_TYPE_SUM_THRESH> : public SCAMPStrategy {
 public:
  __device__ WriteBackStrategy() {}
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       SMEM_TYPE &smem,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    int global_position, local_position;
    if (COMPUTE_COLS) {
      global_position = tile_start_x + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_WIDTH && global_position < n_x) {
        atomicAdd(profile_A + global_position, smem.get_mp_col(local_position));
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
    if (COMPUTE_ROWS) {
      global_position = tile_start_y + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_HEIGHT && global_position < n_y) {
        atomicAdd(profile_B + global_position, smem.get_mp_row(local_position));
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
  }
};

template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ, typename SMEM_TYPE>
class WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS,
                        TILE_WIDTH, TILE_HEIGHT, BLOCKSZ, SMEM_TYPE,
                        PROFILE_TYPE_1NN_INDEX> : public SCAMPStrategy {
 public:
  __device__ WriteBackStrategy() {}
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       SMEM_TYPE &smem,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    int global_position, local_position;
    if (COMPUTE_COLS) {
      global_position = tile_start_x + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_WIDTH && global_position < n_x) {
        mp_entry e;
        e.ulong = smem.get_mp_col(local_position);
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
        e.ulong = smem.get_mp_row(local_position);
        MPatomicMax(profile_B + global_position, e.floats[0], e.ints[1]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
  }
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
template <typename DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class DoIterationStrategy : public SCAMPStrategy {
 public:
  __device__ inline void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                              SMEM_TYPE &smem,
                              OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoIterationStrategy() {}
};

template <typename DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE>
class DoIterationStrategy<DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                          PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                          COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE_SUM_THRESH>
    : public SCAMPStrategy {
 public:
  __device__ DoIterationStrategy() {}
  __device__ void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                              SMEM_TYPE &smem,
                              OptionalArgs &args) {
    DISTANCE_TYPE distc1 = 0;
    DISTANCE_TYPE distc2 = 0;
    DISTANCE_TYPE distc3 = 0;
    DISTANCE_TYPE distc4 = 0;
    DISTANCE_TYPE distc5 = 0;
    DISTANCE_TYPE distc6 = 0;
    DISTANCE_TYPE distc7 = 0;

    // Load row values 2 at a time, load column values 4 at a time
    int r = info.local_row >> 2;
    int c = info.local_col >> 2;

    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a double4 vector type
    VEC4_DATA_TYPE dfc = smem.get_col_vec4(c, smem.df_col_offset);
    VEC4_DATA_TYPE dgc = smem.get_col_vec4(c, smem.dg_col_offset);
    VEC4_DATA_TYPE inormc = smem.get_col_vec4(c, smem.inorm_col_offset);
    VEC4_DATA_TYPE dfc2 = smem.get_col_vec4(c + 1, smem.df_col_offset);
    VEC4_DATA_TYPE dgc2 = smem.get_col_vec4(c + 1, smem.dg_col_offset);
    VEC4_DATA_TYPE inormc2 = smem.get_col_vec4(c + 1, smem.inorm_col_offset);

    // Due to a lack of registers on volta, we only load these row values 2 at a
    // time
    VEC4_DATA_TYPE dgr = smem.get_row_vec4(r, smem.dg_row_offset);
    VEC4_DATA_TYPE dfr = smem.get_row_vec4(r, smem.df_row_offset);
    VEC4_DATA_TYPE inormr = smem.get_row_vec4(r, smem.inorm_row_offset);

    // Do rows one at a time:
    _do_row.exec(info, distc1, distc2, distc3, distc4, inormc.x, inormc.y,
                 inormc.z, inormc.w, inormr.x, dfc.x, dfc.y, dfc.z, dfc.w,
                 dgc.x, dgc.y, dgc.z, dgc.w, dfr.x, dgr.x, smem, args);

    _do_row.exec(info, distc2, distc3, distc4, distc5, inormc.y, inormc.z,
                 inormc.w, inormc2.x, inormr.y, dfc.y, dfc.z, dfc.w, dfc2.x,
                 dgc.y, dgc.z, dgc.w, dgc2.x, dfr.y, dgr.y, smem, args);

    _do_row.exec(info, distc3, distc4, distc5, distc6, inormc.z, inormc.w,
                 inormc2.x, inormc2.y, inormr.z, dfc.z, dfc.w, dfc2.x, dfc2.y,
                 dgc.z, dgc.w, dgc2.x, dgc2.y, dfr.z, dgr.z, smem, args);

    _do_row.exec(info, distc4, distc5, distc6, distc7, inormc.w, inormc2.x,
                 inormc2.y, inormc2.z, inormr.w, dfc.w, dfc2.x, dfc2.y, dfc2.z,
                 dgc.w, dgc2.x, dgc2.y, dgc2.z, dfr.w, dgr.w, smem, args);

    if (COMPUTE_COLS) {
      _update_cols.exec(distc1, distc2, distc3, distc4, distc5, distc6, distc7,
                        smem, info.local_col - DIAGS_PER_THREAD);
    }
    info.global_col += DIAGS_PER_THREAD;
    info.global_row += DIAGS_PER_THREAD;
  }

 private:
  DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                   COMPUTE_ROWS, COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE_SUM_THRESH>
      _do_row;
  UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE, SMEM_TYPE,
                        PROFILE_TYPE_SUM_THRESH>
      _update_cols;
};

template <typename DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE>
class DoIterationStrategy<DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                          PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                          COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE_1NN_INDEX>
    : public SCAMPStrategy {
 public:
  __device__ DoIterationStrategy() {}
  __device__ void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                              SMEM_TYPE &smem,
                              OptionalArgs &args) {
    int lane = threadIdx.x & 0x1f;
    float4 distc = {CC_MIN, CC_MIN, CC_MIN, CC_MIN};
    float4 distc2 = {CC_MIN, CC_MIN, CC_MIN, CC_MIN};
    uint4 idxc, idxc2;

    int r = info.local_row >> 2;
    int c = info.local_col >> 2;

    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a double4 vector type
    VEC4_DATA_TYPE dfc = smem.get_col_vec4(c, smem.df_col_offset);
    VEC4_DATA_TYPE dgc = smem.get_col_vec4(c, smem.dg_col_offset);
    VEC4_DATA_TYPE inormc = smem.get_col_vec4(c, smem.inorm_col_offset);
    VEC4_DATA_TYPE dfc2 = smem.get_col_vec4(c + 1, smem.df_col_offset);
    VEC4_DATA_TYPE dgc2 = smem.get_col_vec4(c + 1, smem.dg_col_offset);
    VEC4_DATA_TYPE inormc2 = smem.get_col_vec4(c + 1, smem.inorm_col_offset);
    ulonglong4 mp_row_check;

    if (COMPUTE_ROWS) {
      mp_row_check = smem.get_mp_row_vec4(r);
      //mp_row_check = reinterpret_cast<ulonglong4 *>(smem.local_mp_row)[r];
    }

    VEC4_DATA_TYPE dgr = smem.get_row_vec4(r, smem.dg_row_offset);
    VEC4_DATA_TYPE dfr = smem.get_row_vec4(r, smem.df_row_offset);
    VEC4_DATA_TYPE inormr = smem.get_row_vec4(r, smem.inorm_row_offset);

    mp_entry e;
    e.ulong = mp_row_check.x;
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

    if (COMPUTE_COLS) {
      ulonglong4 mp_col_check1, mp_col_check2;
      mp_col_check1 = smem.get_mp_col_vec4(c);
      mp_col_check2 = smem.get_mp_col_vec4(c + 1);
      e.ulong = mp_col_check1.x;
      MPatomicMax_check(smem.get_mp_col_addr(info.local_col - 4),
                        distc.x, idxc.x, e.floats[0]);
      e.ulong = mp_col_check1.y;
      MPatomicMax_check(smem.get_mp_col_addr(info.local_col - 3),
                        distc.y, idxc.y, e.floats[0]);
      e.ulong = mp_col_check1.z;
      MPatomicMax_check(smem.get_mp_col_addr(info.local_col - 2),
                        distc.z, idxc.z, e.floats[0]);
      e.ulong = mp_col_check1.w;
      MPatomicMax_check(smem.get_mp_col_addr(info.local_col - 1),
                        distc.w, idxc.w, e.floats[0]);
      e.ulong = mp_col_check2.x;
      MPatomicMax_check(smem.get_mp_col_addr(info.local_col),
                        distc2.x, idxc2.x, e.floats[0]);
      e.ulong = mp_col_check2.y;
      MPatomicMax_check(smem.get_mp_col_addr(info.local_col + 1),
                        distc2.y, idxc2.y, e.floats[0]);
      e.ulong = mp_col_check2.z;
      MPatomicMax_check(smem.get_mp_col_addr(info.local_col + 2),
                        distc2.z, idxc2.z, e.floats[0]);
    }
  }

 private:
  DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, float,
                   COMPUTE_ROWS, COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE_1NN_INDEX>
      _do_row;
  // UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE,
  // PROFILE_TYPE_1NN_INDEX>
  //    _update_cols;
};

///////////////////////////////////////
// Slow path (edge tiles)
// Does a single iteration of the inner loop on 4 diagonals per thread, not
// unrolled Checks for the boundary case where only 1, 2, or 3 diagonals can be
// updated
//////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
//
//  SCAMP TACTIC DESCRIBES STRATEGY FOR WHAT OPS TO EXECUTE IN THE KERNEL
//
/////////////////////////////////////////////////////////////////////////////////////

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ, typename SMEM_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class SCAMPTactic {
 public:
  __device__ SCAMPTactic() {}
  __device__ void InitMem(SCAMPKernelInputArgs<double> &args,
                          SMEM_TYPE &smem,
                          PROFILE_DATA_TYPE *__restrict__ profile_a,
                          PROFILE_DATA_TYPE *__restrict__ profile_b,
                          uint32_t col_start, uint32_t row_start) {
    _init_mem.exec(args, smem, profile_a, profile_b, col_start, row_start);
  }
  __device__ inline __attribute__((always_inline)) void DoIteration(
      SCAMPThreadInfo<ACCUM_TYPE> &info,
      SMEM_TYPE &smem, OptionalArgs &args) {
    _do_iteration.exec(info, smem, args);
  }
  __device__ inline void DoEdge(int i, int j, int x, int y, int n,
                                ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                                ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                                size_t num_diags,
                                SMEM_TYPE &smem,
                                OptionalArgs &args) {
    _do_edge.exec(i, j, x, y, n, cov1, cov2, cov3, cov4, diag, num_diags, smem,
                  args);
  }
  __device__ inline void WriteBack(uint32_t tile_start_x, uint32_t tile_start_y,
                                   uint32_t n_x, uint32_t n_y,
                                   SMEM_TYPE &smem,
                                   PROFILE_DATA_TYPE *__restrict__ profile_A,
                                   PROFILE_DATA_TYPE *__restrict__ profile_B) {
    _do_writeback.exec(tile_start_x, tile_start_y, n_x, n_y, smem, profile_A, profile_B);
  }

 private:
  InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                  TILE_WIDTH, TILE_HEIGHT, BLOCKSZ, SMEM_TYPE, PROFILE_TYPE>
      _init_mem;
  DoIterationStrategy<DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE, PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                      COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE>
      _do_iteration;
  DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                    COMPUTE_ROWS, COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE>
      _do_edge;
  WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS, TILE_WIDTH,
                    TILE_HEIGHT, BLOCKSZ, SMEM_TYPE, PROFILE_TYPE>
      _do_writeback;
};

// Computes the matrix profile given the sliding dot products for the first
// query and the precomputed data statisics
template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE, typename VEC2_PROFILE_TYPE, typename VEC4_PROFILE_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE, int blocks_per_sm, int tile_height,
          int BLOCKSZ>
__global__ void __launch_bounds__(BLOCKSZ, blocks_per_sm)
    do_tile(SCAMPKernelInputArgs<double> args,
            PROFILE_DATA_TYPE *__restrict__ profile_A,
            PROFILE_DATA_TYPE *__restrict__ profile_B) {
  constexpr int diags_per_thread = 4;
  constexpr int tile_width = tile_height + BLOCKSZ * diags_per_thread;
  SCAMPThreadInfo<ACCUM_TYPE> thread_info;
  extern __shared__ char smem_raw[];
  SCAMPSmem<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE> smem(smem_raw, COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height);
  SCAMPTactic<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, PROFILE_DATA_TYPE,
              ACCUM_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, tile_width,
              tile_height, BLOCKSZ, decltype(smem), PROFILE_TYPE>
      tactic;

  const unsigned int start_diag = (threadIdx.x * diags_per_thread) +
                                  blockIdx.x * (blockDim.x * diags_per_thread);

  // This is the index of the meta-diagonal that this thread block will work on
  const unsigned int meta_diagonal_idx = blockIdx.x;

  // The first threads are acutally computing the trivial match between the same
  // subsequence we exclude these from the calculation
  uint32_t tile_start_col =
      meta_diagonal_idx * (BLOCKSZ * diags_per_thread) + args.exclusion_lower;
  uint32_t tile_start_row = 0;

  // x is the global column of the distance matrix
  // y is the global row of the distance matrix
  // localX, localY are the local coordinates of the thread position in the tile
  // it is working on
  thread_info.global_col = tile_start_col + threadIdx.x * diags_per_thread;
  thread_info.global_row = 0;

  const unsigned int num_diags = args.n_x - args.exclusion_upper;

  // Load the first dot product values
  if (thread_info.global_col < args.n_x) {
    thread_info.cov1 = args.cov[thread_info.global_col];
  }

  if (thread_info.global_col + 1 < args.n_x) {
    thread_info.cov2 = args.cov[thread_info.global_col + 1];
  }

  if (thread_info.global_col + 2 < args.n_x) {
    thread_info.cov3 = args.cov[thread_info.global_col + 2];
  }

  if (thread_info.global_col + 3 < args.n_x) {
    thread_info.cov4 = args.cov[thread_info.global_col + 3];
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
  while (tile_start_col < args.n_x && tile_start_row < args.n_y) {
    // Initialize the next tile's shared memory
    tactic.InitMem(args, smem, profile_A, profile_B, tile_start_col,
                   tile_start_row);
    thread_info.local_col = threadIdx.x * diags_per_thread;
    thread_info.local_row = 0;
    // Start of new tile, sync
    __syncthreads();

    // There are 2 pathways here, most of the time we take the fast path (top),
    // the last block (edge_tile) will take the slower path (bottom)
    if (tile_start_col + tile_width < args.n_x &&
        tile_start_row + tile_height < args.n_y &&
        start_diag + diags_per_thread - 1 < num_diags) {
      while (thread_info.local_row < tile_height) {
        tactic.DoIteration(thread_info, smem, args.opt);
      }
      //      thread_info.global_row += tile_height;
      //      thread_info.global_col += tile_height;

    } else if (start_diag < num_diags) {
      while (thread_info.global_col < args.n_x &&
             thread_info.global_row < args.n_y &&
             thread_info.local_row < tile_height) {
        tactic.DoEdge(thread_info.local_row, thread_info.local_col,
                      thread_info.global_col, thread_info.global_row, args.n_x,
                      thread_info.cov1, thread_info.cov2, thread_info.cov3,
                      thread_info.cov4, start_diag, num_diags, smem, args.opt);

        ++thread_info.global_col;
        ++thread_info.global_row;
        ++thread_info.local_col;
        ++thread_info.local_row;
      }
    }

    // After this sync, the caches will be updated with the best so far values
    // for this tile
    __syncthreads();

    tactic.WriteBack(tile_start_col, tile_start_row, args.n_x, args.n_y,
                     smem, profile_A,
                     profile_B);

    // Update the tile position
    tile_start_col += tile_height;
    tile_start_row += tile_height;

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

int get_exclusion(uint64_t window_size, uint64_t start_row,
                  uint64_t start_column) {
  int exclusion = window_size / 4;
  if (!(start_column >= start_row && start_column <= start_row + exclusion)) {
    return 0;
  }
  return window_size / 4;
}

constexpr int FPTypeSize(SCAMPPrecisionType dtype) {
  switch (dtype) {
    case PRECISION_DOUBLE:
      return sizeof(double);
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      return sizeof(float);
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

int get_smem(bool computing_rows, bool computing_cols, int blocksz,
             SCAMPPrecisionType intermediate_data_type, int profile_data_size) {
  constexpr int diags_per_thread = 4;
  constexpr int num_shared_variables = 3;
  int intermediate_data_size = FPTypeSize(intermediate_data_type);
  int tile_height = GetTileHeight(intermediate_data_type);
  int tile_width = blocksz * diags_per_thread + tile_height;
  int smem = (tile_width + tile_height) * num_shared_variables *
             intermediate_data_size;
  std::cout << "shared_elems = " <<  (tile_width + tile_height) * num_shared_variables << std::endl;
  if (computing_cols) {
    smem += tile_width * profile_data_size;
  }
  if (computing_rows) {
    smem += tile_height * profile_data_size;
  }
  return smem;
}

template <typename PROFILE_DATA_TYPE, typename VEC2_PROFILE_TYPE, typename VEC4_PROFILE_TYPE, SCAMPProfileType PROFILE_TYPE,
          int BLOCKSPERSM>
SCAMPError_t LaunchDoTile(SCAMPKernelInputArgs<double> args,
                          PROFILE_DATA_TYPE *profile_A,
                          PROFILE_DATA_TYPE *profile_B,
                          SCAMPPrecisionType fp_type, bool computing_rows,
                          bool computing_cols, uint64_t blocksz,
                          uint64_t num_blocks, uint64_t smem, cudaStream_t s) {
  dim3 block(blocksz, 1, 1);
  dim3 grid(num_blocks, 1, 1);
  if (computing_rows && computing_cols) {
    constexpr bool COMPUTE_COLS = true;
    constexpr bool COMPUTE_ROWS = true;
    switch (fp_type) {
      case PRECISION_DOUBLE: {
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        do_tile<double, double2, double4, double, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, double,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }

      default:
        return SCAMP_CUDA_ERROR;
    }
    return SCAMP_NO_ERROR;
  } else if (computing_cols) {
    constexpr bool COMPUTE_COLS = true;
    constexpr bool COMPUTE_ROWS = false;
    switch (fp_type) {
      case PRECISION_DOUBLE: {
        do_tile<double, double2, double4, double, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, double,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      default:
        return SCAMP_CUDA_ERROR;
    }
  } else if (computing_rows) {
    constexpr bool COMPUTE_COLS = false;
    constexpr bool COMPUTE_ROWS = true;
    switch (fp_type) {
      case PRECISION_DOUBLE: {
        do_tile<double, double2, double4, double, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, double,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      default:
        return SCAMP_CUDA_ERROR;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t kernel_self_join_upper(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    const cudaDeviceProp &props, SCAMPPrecisionType t, const OptionalArgs &args,
    SCAMPProfileType profile_type, cudaStream_t s) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  int32_t exclusion = get_exclusion(window_size, global_col, global_row);
  uint64_t num_workers =
      ceil((tile_width - exclusion) / (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(QT, df_A, df_B, dg_A, dg_B, norms_A,
                                         norms_B, tile_width, tile_height,
                                         exclusion, 0, args);
  uint64_t smem =
      get_smem(true, true, blocksz, t, GetProfileTypeSize(profile_type));
  if (exclusion < tile_width) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, double2, double4, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, true, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, ulonglong2, ulonglong4, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<uint64_t *>(profile_A->at(PROFILE_TYPE_1NN_INDEX)),
            reinterpret_cast<uint64_t *>(profile_B->at(PROFILE_TYPE_1NN_INDEX)),
            t, true, true, blocksz, num_blocks, smem, s);
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
    size_t window_size, size_t tile_width, size_t tile_height,
    size_t global_col, size_t global_row, const cudaDeviceProp &props,
    SCAMPPrecisionType t, const OptionalArgs &args,
    SCAMPProfileType profile_type, cudaStream_t s) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  uint64_t exclusion =
      get_exclusion(window_size, global_col, global_row + tile_height);
  uint64_t num_workers =
      ceil((tile_height - exclusion) / (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(QT, df_B, df_A, dg_B, dg_A, norms_B,
                                         norms_A, tile_height, tile_width, 0,
                                         exclusion, args);
  uint64_t smem =
      get_smem(true, true, blocksz, t, GetProfileTypeSize(profile_type));
  if (exclusion < tile_height) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, double2, double4, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, true, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, ulonglong2, ulonglong4, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<uint64_t *>(profile_B->at(PROFILE_TYPE_1NN_INDEX)),
            reinterpret_cast<uint64_t *>(profile_A->at(PROFILE_TYPE_1NN_INDEX)),
            t, true, true, blocksz, num_blocks, smem, s);
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}
SCAMPError_t kernel_ab_join_upper(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    int64_t distributed_col, int64_t distributed_row,
    const cudaDeviceProp &props, SCAMPPrecisionType t, bool computing_rows,
    const OptionalArgs &args, SCAMPProfileType profile_type, cudaStream_t s) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  int32_t exclusion;
  std::cout << distributed_col << " distributed " << distributed_row
            << std::endl;
  if (distributed_col < 0 || distributed_row < 0) {
    exclusion = 0;
  } else {
    exclusion = get_exclusion(window_size, global_col + distributed_col,
                              global_row + distributed_row);
  }
  uint64_t num_workers =
      ceil((tile_width - exclusion) / (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(QT, df_A, df_B, dg_A, dg_B, norms_A,
                                         norms_B, tile_width, tile_height,
                                         exclusion, 0, args);
  uint64_t smem = get_smem(computing_rows, true, blocksz, t,
                           GetProfileTypeSize(profile_type));
  std::cout << "num_workers " << num_workers << " num_blocks " << num_blocks
            << " blocksz " << blocksz << " smem (KB)" << smem / 1024
            << std::endl;
  std::cout << "Exclusion = " << exclusion << std::endl;
  if (exclusion < tile_width) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, double2, double4, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, computing_rows, true, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, ulonglong2, ulonglong4, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<uint64_t *>(profile_A->at(PROFILE_TYPE_1NN_INDEX)),
            reinterpret_cast<uint64_t *>(profile_B->at(PROFILE_TYPE_1NN_INDEX)),
            t, computing_rows, true, blocksz, num_blocks, smem, s);
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t kernel_ab_join_lower(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    int64_t distributed_col, int64_t distributed_row,
    const cudaDeviceProp &props, SCAMPPrecisionType t, bool computing_rows,
    const OptionalArgs &args, SCAMPProfileType profile_type, cudaStream_t s) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  int32_t exclusion;
  if (distributed_col < 0 || distributed_row < 0) {
    exclusion = 0;
  } else {
    exclusion = get_exclusion(window_size, global_col + distributed_col,
                              global_row + distributed_row + tile_height);
  }
  std::cout << "Exclusion = " << exclusion << std::endl;
  uint64_t num_workers =
      ceil((tile_height - exclusion) / (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(QT, df_B, df_A, dg_B, dg_A, norms_B,
                                         norms_A, tile_height, tile_width, 0,
                                         exclusion, args);
  uint64_t smem = get_smem(computing_rows, true, blocksz, t,
                           GetProfileTypeSize(profile_type));
  if (exclusion < tile_height) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, double2, double4, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, computing_rows, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, ulonglong2, ulonglong4, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<uint64_t *>(profile_B->at(PROFILE_TYPE_1NN_INDEX)),
            reinterpret_cast<uint64_t *>(profile_A->at(PROFILE_TYPE_1NN_INDEX)),
            t, true, computing_rows, blocksz, num_blocks, smem, s);
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
