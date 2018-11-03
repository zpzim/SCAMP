#include <unordered_map>
#include "kernels.h"

namespace SCAMP {

#define EXTRA_REGISTERS_PER_THREAD 4
constexpr int DIAGS_PER_THREAD = 2;
constexpr int BLOCKSZ_SP = 512;
constexpr int BLOCKSZ_DP = 256;
constexpr int BLOCKSPERSM_SELF = 6;
constexpr int BLOCKSPERSM_AB = 6;
constexpr int TILE_HEIGHT_SP = 32 * EXTRA_REGISTERS_PER_THREAD;
constexpr int TILE_HEIGHT_DP = 32 * EXTRA_REGISTERS_PER_THREAD;
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
        _input_data_cols_a = nullptr;
        _input_data_cols_b = nullptr;
        _input_data_cols = nullptr;
        _input_data_rows = (DATA_TYPE*) smem;
    } else {
        _input_data_cols_a = nullptr;
        _input_data_cols_b = nullptr;
        _input_data_cols = nullptr;
        _input_data_rows = (DATA_TYPE*) smem;
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
  __device__ inline VEC2_DATA_TYPE get_col_vec2(int chunk, int offset) {
    bool odd = chunk & 1;
    if (sizeof(DATA_TYPE) == 8) {
        chunk = chunk >> 1;
        if (odd) {
            return reinterpret_cast<VEC2_DATA_TYPE*>(_input_data_cols_b + (chunk * stride + offset))[0];
        }
        return reinterpret_cast<VEC2_DATA_TYPE*>(_input_data_cols_a + (chunk * stride + offset))[0];
    }
    return reinterpret_cast<VEC2_DATA_TYPE*>(_input_data_cols + (chunk * stride + 2 * offset))[0];
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
  __device__ inline VEC2_PROFILE_TYPE get_mp_col_vec2(int chunk) {
    bool odd = chunk & 1;
    if (sizeof(DATA_TYPE) == 8) {
        chunk = chunk >> 1;
        if (odd) {
            return reinterpret_cast<VEC2_PROFILE_TYPE*>(_mp_col_b + (chunk * 2))[0];
        }
        return reinterpret_cast<VEC2_PROFILE_TYPE*>(_mp_col_a + (chunk * 2))[0];
    }
    return reinterpret_cast<VEC2_PROFILE_TYPE*>(_mp_col + (chunk * 2))[0];
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
  bool init_opt_args;
  bool final_opt_iter;
  int lane;
  int warp_id;
  int mask1, mask2;
  float4 distc;
  float distc_extra;
  uint4 idxc;
  uint32_t idxc_extra;
  double4 inormc, dgc, dfc;
  double inormc_extra, dgc_extra, dfc_extra;
  double inormc_cache[EXTRA_REGISTERS_PER_THREAD];  
  double dgc_cache[EXTRA_REGISTERS_PER_THREAD];  
  double dfc_cache[EXTRA_REGISTERS_PER_THREAD];  
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

// Atomically updates the MP/idxs using a single 64-bit integer. We lose a small
// amount of precision in the output, if we do not do this we are unable
// to atomically update both the matrix profile and the indexes without using a
// critical section and dedicated locks.
__device__ inline void MPatomicMax_block(volatile uint64_t *address, float val,
                                   unsigned int idx) {
  mp_entry loc, loctest;
  loc.floats[0] = val;
  loc.ints[1] = idx;
  loctest.ulong = *address;
  while (loctest.floats[0] < val) {
    loctest.ulong =
        atomicCAS_block((unsigned long long int *)address, loctest.ulong, loc.ulong);
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
      loctest.ulong = atomicCAS_block((unsigned long long int *)address,
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
                       SCAMPThreadInfo<double> &info,
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
                               SCAMPThreadInfo<double> &info,
                               PROFILE_DATA_TYPE *__restrict__ profile_a,
                               PROFILE_DATA_TYPE *__restrict__ profile_b,
                               uint32_t col_start, uint32_t row_start) {
    int global_position = col_start + threadIdx.x * DIAGS_PER_THREAD;
    int local_position = threadIdx.x;
    int next_warp_position = col_start + 32*DIAGS_PER_THREAD*(info.warp_id + 1) + info.lane;
    if (global_position < args.n_x) {
      info.dgc.x = args.dga[global_position];
      info.dfc.x = args.dfa[global_position];
      info.inormc.x = args.normsa[global_position];
   } 
   if (global_position + 1 < args.n_x) { 
      info.dgc.y = args.dga[global_position + 1];
      info.dfc.y = args.dfa[global_position + 1];
      info.inormc.y = args.normsa[global_position + 1];
    }
    

    if (global_position + 2 < args.n_x) { 
      info.dgc.z = args.dga[global_position + 2];
      info.dfc.z = args.dfa[global_position + 2];
      info.inormc.z = args.normsa[global_position + 2];
    }


    if (global_position + 3 < args.n_x) { 
      info.dgc.w = args.dga[global_position + 3];
      info.dfc.w = args.dfa[global_position + 3];
      info.inormc.w = args.normsa[global_position + 3];
    }


    if (global_position + 4 < args.n_x) { 
      info.dgc_extra = args.dga[global_position + 4];
      info.dfc_extra = args.dfa[global_position + 4];
      info.inormc_extra = args.normsa[global_position + 4];
    }
      for (int i = 0; i < EXTRA_REGISTERS_PER_THREAD && next_warp_position < args.n_x; ++i) {
        info.inormc_cache[i] = args.normsa[next_warp_position];
        info.dgc_cache[i] = args.dga[next_warp_position];
        info.dfc_cache[i] = args.dfa[next_warp_position];
        next_warp_position += 32;
      }
//      if (warpId > info.warpId && warpId <= info.warpId + EXTRA_REGISTERS_PER_THREAD)
//      smem.set_col(local_position, args.dga[global_position], smem.dg_col_offset);
//      smem.set_col(local_position, args.dfa[global_position], smem.df_col_offset);
//      smem.set_col(local_position, args.normsa[global_position], smem.inorm_col_offset);

    global_position = col_start + threadIdx.x;
    while (local_position < tile_width && global_position < args.n_x) {
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
/*
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
*/


template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS, typename SMEM_TYPE>
class DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                       COMPUTE_ROWS, COMPUTE_COLS, SMEM_TYPE, PROFILE_TYPE_1NN_INDEX> {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE &distc1,
      DISTANCE_TYPE &distc2,  uint32_t &idxc1, uint32_t &idxc2,
      const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, float &curr_mp_row_val,
      SMEM_TYPE &smem, OptionalArgs &args) {
    DISTANCE_TYPE distx = static_cast<DATA_TYPE>(info.cov1) * inormcx * inormr;
    DISTANCE_TYPE disty = static_cast<DATA_TYPE>(info.cov2) * inormcy * inormr;
    // Compute the next covariance values
    info.cov1 = info.cov1 + df_colx * dg_row + dg_colx * df_row;
    info.cov2 = info.cov2 + df_coly * dg_row + dg_coly * df_row;
    // Update the column best-so-far values

    if (COMPUTE_COLS) {
      MPMax2<DISTANCE_TYPE>(distc1, distx, idxc1, info.global_row);
      MPMax2<DISTANCE_TYPE>(distc2, disty, idxc2, info.global_row);
    }

    if (COMPUTE_ROWS) {
      // We take the maximum of the columns we computed for the row
      // And use that value to check the matrix profile
      uint32_t idx;
      DISTANCE_TYPE d;
      if (distx > disty) {
        d = distx;
        idx = info.global_col;
      } else {
        d = disty;
        idx = info.global_col + 1;
      }
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
  __device__ inline void exec(SCAMPThreadInfo<ACCUM_TYPE> &info, int i, int j, int x, int y, int n,
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
  __device__ inline void exec(SCAMPThreadInfo<ACCUM_TYPE> &info, int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SMEM_TYPE &smem,
                              OptionalArgs &args) {
    float dist_row;
    uint32_t idx_row;
    float distx;
    float disty;
//    float distz;
//    float distw;

    DATA_TYPE inormr = smem.get_row(i, smem.inorm_row_offset );
    DATA_TYPE dgr = smem.get_row(i, smem.dg_row_offset);
    DATA_TYPE dfr = smem.get_row(i, smem.df_row_offset);

    info.inormc.x = info.inormc.y;
    info.inormc.y = info.inormc.z;
    info.inormc.z = info.inormc.w;
    info.inormc.w = info.inormc_extra;

    info.dfc.x = info.dfc.y;
    info.dfc.y = info.dfc.z;
    info.dfc.z = info.dfc.w;
    info.dfc.w = info.dfc_extra;
    
    info.dgc.x = info.dgc.y;
    info.dgc.y = info.dgc.z;
    info.dgc.z = info.dgc.w;
    info.dgc.w = info.dgc_extra;
/*
    int k = (info.local_row - 1)>> 5;
    int lane = (info.local_row - 1) & 31;
    int curr_mask = 0;
    int other_mask = 0;
    
      if (info.lane == 30 || info.lane == 31 || info.lane == lane) {
        curr_mask = 1 << (31 - info.lane);
        other_mask = 1 << (31 - lane);  
      }
      
      DATA_TYPE dfcy_val = __shfl_sync(curr_mask | other_mask, info.dfc_cache[k], lane);
      DATA_TYPE dgcy_val = __shfl_sync(curr_mask | other_mask, info.dgc_cache[k], lane);
      DATA_TYPE inormcy_val = __shfl_sync(curr_mask | other_mask, info.inormc_cache[k], lane);
      lane++;
      if (lane == 32) {
        lane = 0;
        k++;
      }
      other_mask = 1 << (31 - lane);
      DATA_TYPE dfcz_val = __shfl_sync(curr_mask | other_mask, info.dfc_cache[k], lane);
      DATA_TYPE dgcz_val = __shfl_sync(curr_mask | other_mask, info.dgc_cache[k], lane);
      DATA_TYPE inormcz_val = __shfl_sync(curr_mask | other_mask, info.inormc_cache[k], lane);
      lane++;
      if (lane == 32) {
        lane = 0;
        k++;
      }
      other_mask = 1 << (31 - lane);
      DATA_TYPE dfcw_val = __shfl_sync(curr_mask | other_mask, info.dfc_cache[k], lane);
      DATA_TYPE dgcw_val = __shfl_sync(curr_mask | other_mask, info.dgc_cache[k], lane);
      DATA_TYPE inormcw_val = __shfl_sync(curr_mask | other_mask, info.inormc_cache[k], lane);
      lane++;
      if (lane == 32) {
        lane = 0;
        k++;
      }
      other_mask = 1 << (31 - lane);
      DATA_TYPE dfcex_val = __shfl_sync(curr_mask | other_mask, info.dfc_cache[k], lane);
      DATA_TYPE dgcex_val = __shfl_sync(curr_mask | other_mask, info.dgc_cache[k], lane);
      DATA_TYPE inormcex_val = __shfl_sync(curr_mask | other_mask, info.inormc_cache[k], lane);
      lane++;
      if (lane == 32) {
        lane = 0;
        k++;
      }

      if (info.lane == 30) {
        info.dfc.w = dfcy_val;
        info.dfc_extra = dfcz_val;
        info.dgc.w = dgcy_val;
        info.dgc_extra = dgcz_val;
        info.inormc.w = inormcy_val;
        info.inormc_extra = inormcz_val;
//        info.dfc.w = smem.get_col(info.local_col + 3, smem.df_col_offset);
//        info.dfc_extra = smem.get_col(info.local_col + 4, smem.df_col_offset);
//        info.dgc.w = smem.get_col(info.local_col + 3, smem.dg_col_offset);
//        info.dgc_extra = smem.get_col(info.local_col + 4, smem.dg_col_offset);
//        info.inormc.w = smem.get_col(info.local_col + 3, smem.inorm_col_offset);
//        info.inormc_extra = smem.get_col(info.local_col + 4, smem.inorm_col_offset);
      }
      __syncwarp(); 
      if (info.lane == 31) {
        info.dfc.y = dfcy_val;
        info.dfc.z = dfcz_val;
        info.dgc.y = dgcy_val;
        info.dgc.z = dgcz_val;
        info.inormc.y = inormcy_val;
        info.inormc.z = inormcz_val;
        info.dfc.w = dfcw_val;
        info.dfc_extra = dfcex_val;
        info.dgc.w = dgcw_val;
        info.dgc_extra = dgcex_val;
        info.inormc.w = inormcw_val;
        info.inormc_extra = inormcex_val;
       } 
*/


    // Compute the next set of distances (row y)
    //distx = cov1 * smem.get_col(j, smem.inorm_col_offset) * inormr;
    //disty = cov2 * smem.get_col(j + 1, smem.inorm_col_offset) * inormr;
    distx = cov1 * info.inormc.x * inormr;
    disty = cov2 * info.inormc.y * inormr;
//    distz = cov3 * smem.get_col(j + 2, smem.inorm_col_offset) * inormr;
//    distw = cov4 * smem.get_col(j + 3, smem.inorm_col_offset) * inormr;

    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + info.dfc.x * dgr + info.dgc.x * dfr;
    cov2 = cov2 + info.dfc.y * dgr + info.dgc.y * dfr;
//    cov3 = cov3 + smem.get_col(j + 2, smem.df_col_offset) * dgr + smem.get_col(j + 2, smem.dg_col_offset) * dfr;
//    cov4 = cov4 + smem.get_col(j + 3, smem.df_col_offset) * dgr + smem.get_col(j + 3, smem.dg_col_offset) * dfr;
    
    

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
/*
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
*/
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

template<typename T, typename T4>
__device__ inline void ShuffleData(int mask1, int mask2, T& send, T& send_recv, T4& recv) {
      T temp1 = __shfl_down_sync(mask1, send, 1);
      T temp2 = __shfl_down_sync(mask1, send_recv, 1);
      T temp3 = __shfl_down_sync(mask2, send, 2);
      T temp4 = __shfl_down_sync(mask2, send_recv, 2);
      recv.x = send_recv;
      recv.y = temp1;
      recv.z = temp2;
      recv.w = temp3;
      send_recv = temp4;
}

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
    
    int r = info.local_row >> 2;
    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a double4 vector type
    info.distc = {CC_MIN, CC_MIN, CC_MIN, CC_MIN};
    info.distc_extra = CC_MIN;
    info.idxc = {-1u, -1u, -1u, -1u};
    info.idxc_extra = -1u;

    if (info.init_opt_args) {
      info.init_opt_args = false;
    } else {
      //info.UpdateData(); 
      ShuffleData<DATA_TYPE, VEC4_DATA_TYPE>(info.mask1, info.mask2, info.dgc.w, info.dgc_extra, info.dgc);
      ShuffleData<DATA_TYPE, VEC4_DATA_TYPE>(info.mask1, info.mask2, info.dfc.w, info.dfc_extra, info.dfc);
      ShuffleData<DATA_TYPE, VEC4_DATA_TYPE>(info.mask1, info.mask2, info.inormc.w, info.inormc_extra, info.inormc);
      // TODO(zpzim): There is a method to shuffle the distances along with the rest of that data, taking the max along the way, but I am not sure it is faster.     
      //ShuffleData<float, float4>(info.mask1, info.mask2, info.distc.w, info.distc_extra, info.distc);
      //ShuffleData<uint32_t, uint4>(info.mask1, info.mask2, info.idxc.w, info.idxc_extra, info.idxc);
      int k = (info.local_row - 1)>> 5;
      int lane = (info.local_row - 1) & 31;
      int curr_mask = 0;
      int other_mask = 0;
    
      if (info.lane == 30 || info.lane == 31 || info.lane == lane) {
        curr_mask = 1 << (31 - info.lane);
        other_mask = 1 << (31 - lane);  
      }
      
      DATA_TYPE dfcy_val = __shfl_sync(curr_mask | other_mask, info.dfc_cache[k], lane);
      DATA_TYPE dgcy_val = __shfl_sync(curr_mask | other_mask, info.dgc_cache[k], lane);
      DATA_TYPE inormcy_val = __shfl_sync(curr_mask | other_mask, info.inormc_cache[k], lane);
      lane++;
      if (lane == 32) {
        lane = 0;
        k++;
      }
      other_mask = 1 << (31 - lane);
      DATA_TYPE dfcz_val = __shfl_sync(curr_mask | other_mask, info.dfc_cache[k], lane);
      DATA_TYPE dgcz_val = __shfl_sync(curr_mask | other_mask, info.dgc_cache[k], lane);
      DATA_TYPE inormcz_val = __shfl_sync(curr_mask | other_mask, info.inormc_cache[k], lane);
      lane++;
      if (lane == 32) {
        lane = 0;
        k++;
      }
      other_mask = 1 << (31 - lane);
      DATA_TYPE dfcw_val = __shfl_sync(curr_mask | other_mask, info.dfc_cache[k], lane);
      DATA_TYPE dgcw_val = __shfl_sync(curr_mask | other_mask, info.dgc_cache[k], lane);
      DATA_TYPE inormcw_val = __shfl_sync(curr_mask | other_mask, info.inormc_cache[k], lane);
      lane++;
      if (lane == 32) {
        lane = 0;
        k++;
      }
      other_mask = 1 << (31 - lane);
      DATA_TYPE dfcex_val = __shfl_sync(curr_mask | other_mask, info.dfc_cache[k], lane);
      DATA_TYPE dgcex_val = __shfl_sync(curr_mask | other_mask, info.dgc_cache[k], lane);
      DATA_TYPE inormcex_val = __shfl_sync(curr_mask | other_mask, info.inormc_cache[k], lane);
      lane++;
      if (lane == 32) {
        lane = 0;
        k++;
      }

      if (info.lane == 30) {
        info.distc.w = CC_MIN;
        info.distc_extra = CC_MIN;
        info.idxc.w = -1u;
        info.idxc_extra = -1u;
        info.dfc.w = dfcy_val;
        info.dfc_extra = dfcz_val;
        info.dgc.w = dgcy_val;
        info.dgc_extra = dgcz_val;
        info.inormc.w = inormcy_val;
        info.inormc_extra = inormcz_val;
//        info.dfc.w = smem.get_col(info.local_col + 3, smem.df_col_offset);
//        info.dfc_extra = smem.get_col(info.local_col + 4, smem.df_col_offset);
//        info.dgc.w = smem.get_col(info.local_col + 3, smem.dg_col_offset);
//        info.dgc_extra = smem.get_col(info.local_col + 4, smem.dg_col_offset);
//        info.inormc.w = smem.get_col(info.local_col + 3, smem.inorm_col_offset);
//        info.inormc_extra = smem.get_col(info.local_col + 4, smem.inorm_col_offset);
      }
      __syncwarp(); 
      if (info.lane == 31) {
        info.distc.y = CC_MIN;
        info.distc.z = CC_MIN;
        info.idxc.y = -1u;
        info.idxc.z = -1u;
        info.dfc.y = dfcy_val;
        info.dfc.z = dfcz_val;
        info.dgc.y = dgcy_val;
        info.dgc.z = dgcz_val;
        info.inormc.y = inormcy_val;
        info.inormc.z = inormcz_val;
        info.dfc.w = dfcw_val;
        info.dfc_extra = dfcex_val;
        info.dgc.w = dgcw_val;
        info.dgc_extra = dgcex_val;
        info.inormc.w = inormcw_val;
        info.inormc_extra = inormcex_val;
        
/*
        info.dfc.y = smem.get_col(info.local_col + 1, smem.df_col_offset);
        info.dfc.z = smem.get_col(info.local_col + 2, smem.df_col_offset);
        info.dgc.y = smem.get_col(info.local_col + 1, smem.dg_col_offset);
        info.dgc.z = smem.get_col(info.local_col + 2, smem.dg_col_offset);
        info.inormc.y = smem.get_col(info.local_col + 1, smem.inorm_col_offset);
        info.inormc.z = smem.get_col(info.local_col + 2, smem.inorm_col_offset);
*/
      }
      __syncwarp(); 
      

    }

    
    ulonglong4 mp_row_check;

    if (COMPUTE_ROWS) {
      mp_row_check = smem.get_mp_row_vec4(r);
    }

    VEC4_DATA_TYPE dgr = smem.get_row_vec4(r, smem.dg_row_offset);
    VEC4_DATA_TYPE dfr = smem.get_row_vec4(r, smem.df_row_offset);
    VEC4_DATA_TYPE inormr = smem.get_row_vec4(r, smem.inorm_row_offset);

    mp_entry e;
    e.ulong = mp_row_check.x;
    _do_row.exec(info, info.distc.x, info.distc.y, info.idxc.x, info.idxc.y, info.inormc.x, info.inormc.y, inormr.x, info.dfc.x, info.dfc.y, info.dgc.x, info.dgc.y,  dfr.x, dgr.x, e.floats[0], smem, args);
    e.ulong = mp_row_check.y;
    _do_row.exec(info, info.distc.y, info.distc.z, info.idxc.y, info.idxc.z, info.inormc.y, info.inormc.z, inormr.y, info.dfc.y, info.dfc.z, info.dgc.y, info.dgc.z,  dfr.y, dgr.y, e.floats[0], smem, args);
    e.ulong = mp_row_check.z;
    _do_row.exec(info, info.distc.z, info.distc.w, info.idxc.z, info.idxc.w, info.inormc.z,  info.inormc.w,  inormr.z,  info.dfc.z,  info.dfc.w,  info.dgc.z,  info.dgc.w,  dfr.z, dgr.z, e.floats[0], smem, args);
    e.ulong = mp_row_check.w;
    _do_row.exec(info, info.distc.w, info.distc_extra, info.idxc.w, info.idxc_extra, info.inormc.w, info.inormc_extra, inormr.w, info.dfc.w, info.dfc_extra, info.dgc.w, info.dgc_extra,  dfr.w, dgr.w, e.floats[0], smem, args);
/*
    if (threadIdx.x == 2 && blockIdx.x == 0) {
        printf("row = %d, thread = %d, block = %d\n", info.global_row, threadIdx.x, blockIdx.x);
        printf("row = %d, threadIdx = %d, dx = %lf, dy = %lf, dz = %lf, dw = %lf, dex = %lf\n", info.global_row, threadIdx.x, info.distc.x, info.distc.y, info.distc.z, info.distc.w, info.distc_extra);
        printf("threadIdx = %d, dgx = %lf, dgy = %lf, dgz = %lf, dgw = %lf, dgex = %lf\n", threadIdx.x, info.dgc.x, info.dgc.y, info.dgc.z, info.dgc.w, info.dgc_extra);
        printf("threadIdx = %d, inormx = %lf, inormy = %lf, inormz = %lf, inormw = %lf, inormex = %lf\n", threadIdx.x, info.inormc.x, info.inormc.y, info.inormc.z, info.inormc.w, info.inormc_extra);
        printf("threadIdx = %d, inormr.x = %lf, inormr.y = %lf, inormr.z = %lf, inormr.w = %lf\n", threadIdx.x, inormr.x, inormr.y, inormr.z, inormr.w);
        printf("threadIdx = %d, dgr.x = %lf, dgr.y = %lf, dgr.z = %lf, dgr.w = %lf\n", threadIdx.x, dgr.x, dgr.y, dgr.z, dgr.w);
        printf("threadIdx = %d, dfr.x = %lf, dfr.y = %lf, dfr.z = %lf, dfr.w = %lf\n", threadIdx.x, dfr.x, dfr.y, dfr.z, dfr.w);
        printf("threadIdx = %d, cov1 = %lf, cov2 = %lf\n", threadIdx.x, info.cov1, info.cov2);
    }
*/

    if (COMPUTE_COLS) {
      MPatomicMax_block(smem.get_mp_col_addr(info.local_col - 4),
                           info.distc.x, info.idxc.x);// e.floats[0]);
      MPatomicMax_block(smem.get_mp_col_addr(info.local_col - 3),
                            info.distc.y, info.idxc.y);// e.floats[0]);
      MPatomicMax_block(smem.get_mp_col_addr(info.local_col - 2),
                            info.distc.z, info.idxc.z);// e.floats[0]);
      
      MPatomicMax_block(smem.get_mp_col_addr(info.local_col - 1),
                            info.distc.w, info.idxc.w);//e.floats[0]);
      MPatomicMax_block(smem.get_mp_col_addr(info.local_col), info.distc_extra, info.idxc_extra);
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
                          SCAMPThreadInfo<double> &info,
                          PROFILE_DATA_TYPE *__restrict__ profile_a,
                          PROFILE_DATA_TYPE *__restrict__ profile_b,
                          uint32_t col_start, uint32_t row_start) {
    _init_mem.exec(args, smem, info, profile_a, profile_b, col_start, row_start);
  }
  __device__ inline __attribute__((always_inline)) void DoIteration(
      SCAMPThreadInfo<ACCUM_TYPE> &info,
      SMEM_TYPE &smem, OptionalArgs &args) {
    _do_iteration.exec(info, smem, args);
  }
  __device__ inline void DoEdge(SCAMPThreadInfo<ACCUM_TYPE>& info, int i, int j, int x, int y, int n,
                                ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                                ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                                size_t num_diags,
                                SMEM_TYPE &smem,
                                OptionalArgs &args) {
    _do_edge.exec(info, i, j, x, y, n, cov1, cov2, cov3, cov4, diag, num_diags, smem,
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

  constexpr int diags_per_thread = 2;
  constexpr int tile_width = tile_height + BLOCKSZ * diags_per_thread;
  SCAMPThreadInfo<ACCUM_TYPE> thread_info;
  extern __shared__ char smem_raw[];
  SCAMPSmem<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE> smem(smem_raw, COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height);
  SCAMPTactic<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, PROFILE_DATA_TYPE,
              ACCUM_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, tile_width,
              tile_height, BLOCKSZ, decltype(smem), PROFILE_TYPE>
      tactic;
  thread_info.lane = threadIdx.x & 31;
  thread_info.warp_id = threadIdx.x >> 5;
  thread_info.mask1 = 1 << (31 - thread_info.lane);
  thread_info.mask2 = 1 << (31 - thread_info.lane);
  if (thread_info.lane < 31) {
    thread_info.mask1 |= (thread_info.mask1 >> 1); 
  }
  if (thread_info.lane < 30) {
    thread_info.mask2 |= (thread_info.mask2 >> 2); 
  }
  const uint32_t start_diag = (threadIdx.x * diags_per_thread) +
                                  blockIdx.x * (blockDim.x * diags_per_thread);

  // This is the index of the meta-diagonal that this thread block will work on
  const uint32_t meta_diagonal_idx = blockIdx.x;

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

  const uint32_t num_diags = args.n_x - args.exclusion_upper;
  // Load the first dot product values
  if (thread_info.global_col < args.n_x) {
    thread_info.cov1 = args.cov[thread_info.global_col];
  }

  if (thread_info.global_col + 1 < args.n_x) {
    thread_info.cov2 = args.cov[thread_info.global_col + 1];
  }

/*
  if (thread_info.global_col + 2 < args.n_x) {
    thread_info.cov3 = args.cov[thread_info.global_col + 2];
  }

  if (thread_info.global_col + 3 < args.n_x) {
    thread_info.cov4 = args.cov[thread_info.global_col + 3];
  }
*/
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
    tactic.InitMem(args, smem, thread_info, profile_A, profile_B, tile_start_col,
                   tile_start_row);
    thread_info.local_col = threadIdx.x * diags_per_thread;
    thread_info.local_row = 0;
    // Start of new tile, sync
    __syncthreads();
    //int itcount = 0;
    // There are 2 pathways here, most of the time we take the fast path (top),
    // the last block (edge_tile) will take the slower path (bottom)
    thread_info.init_opt_args = true;
    if (tile_start_col + tile_width < args.n_x &&
        tile_start_row + tile_height < args.n_y &&
        start_diag + diags_per_thread - 1 < num_diags) {
      while (thread_info.local_row < tile_height) {
        //if (thread_info.local_row + 4 >= tile_height) {
        //    thread_info.final_opt_iter = true;
        //}
        tactic.DoIteration(thread_info, smem, args.opt);
      }

    } else if (start_diag < num_diags) {
      while (thread_info.global_col < args.n_x &&
             thread_info.global_row < args.n_y &&
             thread_info.local_row < tile_height) {
     //   tactic.DoEdge(thread_info, thread_info.local_row, thread_info.local_col,
     //                 thread_info.global_col, thread_info.global_row, args.n_x,
     //                 thread_info.cov1, thread_info.cov2, thread_info.cov3,
     //                 thread_info.cov4, start_diag, num_diags, smem, args.opt);
 
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
  return 2;
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
  constexpr int diags_per_thread = 2;
  constexpr int num_shared_variables = 3;
  int intermediate_data_size = FPTypeSize(intermediate_data_type);
  int tile_height = GetTileHeight(intermediate_data_type);
  int tile_width = blocksz * diags_per_thread + tile_height;
  int smem = (tile_height) * num_shared_variables *
             intermediate_data_size;
  smem += tile_width * intermediate_data_size;
  std::cout << "shared_elems = " <<  (tile_width + tile_height) * num_shared_variables << std::endl;
  if (computing_cols) {
    smem += tile_width * profile_data_size;
  }
  if (computing_rows) {
    smem += tile_height * profile_data_size;
  }
  std::cout << "smem = " <<  smem / 1024 << " KB" << std::endl;
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
  printf("blocksz %lu grid size %lu, smem %lu\n", blocksz, num_blocks, smem);
  if (computing_rows && computing_cols) {
    constexpr bool COMPUTE_COLS = true;
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
        //do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
        //        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
        //       TILE_HEIGHT_SP, BLOCKSZ_SP>
        //    <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        //do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
        //        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
        //        TILE_HEIGHT_SP, BLOCKSZ_SP>
        //    <<<grid, block, smem, s>>>(args, profile_A, profile_B);
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
        //do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
        //        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
        //       TILE_HEIGHT_SP, BLOCKSZ_SP>
        //    <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        //do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
        //       COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
        //        TILE_HEIGHT_SP, BLOCKSZ_SP>
        //    <<<grid, block, smem, s>>>(args, profile_A, profile_B);
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
        //do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
        //        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
        //        TILE_HEIGHT_SP, BLOCKSZ_SP>
        //    <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        //do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, VEC2_PROFILE_TYPE, VEC4_PROFILE_TYPE, float,
        //        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
        //        TILE_HEIGHT_SP, BLOCKSZ_SP>
        //    <<<grid, block, smem, s>>>(args, profile_A, profile_B);
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
  constexpr int diags_per_thread = 2;
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
  constexpr int diags_per_thread = 2;
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
  constexpr int diags_per_thread = 2;
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
  constexpr int diags_per_thread = 2;
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
