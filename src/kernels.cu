#include <unordered_map>
#include "defines.h"
#include "kernels.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
// Double atomicAdd is not implemented in hardware before Pascal, providing a
// software implementation here
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

namespace SCAMP {

constexpr int DIAGS_PER_THREAD = 4;
constexpr int BLOCKSZ_SP = 512;
constexpr int BLOCKSZ_DP = 256;
constexpr int BLOCKSPERSM = 2;
constexpr int TILE_HEIGHT_SP = KERNEL_TILE_HEIGHT;
constexpr int TILE_HEIGHT_DP = KERNEL_TILE_HEIGHT;

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

  SCAMPKernelInputArgs(Tile *t, bool transpose, bool ab_join) {
    cov = t->QT();
    dfa = transpose ? t->dfb() : t->dfa();
    dfb = transpose ? t->dfa() : t->dfb();
    dga = transpose ? t->dgb() : t->dga();
    dgb = transpose ? t->dga() : t->dgb();
    normsa = transpose ? t->normsb() : t->normsa();
    normsb = transpose ? t->normsa() : t->normsb();
    n_x = transpose ? t->get_tile_height() : t->get_tile_width();
    n_y = transpose ? t->get_tile_width() : t->get_tile_height();
    n_x = n_x - t->info()->mp_window + 1;
    n_y = n_y - t->info()->mp_window + 1;
    std::pair<int, int> exclusion =
        ab_join ? t->get_exclusion_for_ab_join(!transpose)
                : t->get_exclusion_for_self_join(!transpose);
    exclusion_lower = exclusion.first;
    exclusion_upper = exclusion.second;
    opt = t->info()->opt_args;
    profile_length = t->get_mutable_dev_length();
  }
  const T *__restrict__ cov;
  const T *__restrict__ dfa;
  const T *__restrict__ dfb;
  const T *__restrict__ dga;
  const T *__restrict__ dgb;
  const T *__restrict__ normsa;
  const T *__restrict__ normsb;
  const T *__restrict__ extras[3];
  unsigned long long int *profile_length;
  uint32_t n_x;
  uint32_t n_y;
  int32_t exclusion_lower;
  int32_t exclusion_upper;
  OptionalArgs opt;
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, SCAMPProfileType type>
struct SCAMPSmem {
  __device__ SCAMPSmem(char *smem, bool compute_rows, bool compute_columns,
                       int tile_width, int tile_height, int extra_operands) {
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
    } else {
      local_mp_col = nullptr;
    }
    if (compute_rows) {
      local_mp_row = (PROFILE_DATA_TYPE *)(smem + curr_byte);
      curr_byte += tile_height * profile_size;
    } else {
      local_mp_row = nullptr;
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

enum SCAMPAtomicType { ATOMIC_BLOCK, ATOMIC_GLOBAL, ATOMIC_SYSTEM };

template <typename T, SCAMPAtomicType type>
__device__ inline T do_atomicCAS(T *address, T v1, T v2) {
#if __CUDA_ARCH__ < 600
  return atomicCAS(address, v1, v2);
#else
  switch (type) {
    case ATOMIC_BLOCK:
      return atomicCAS_block(address, v1, v2);
    case ATOMIC_GLOBAL:
      return atomicCAS(address, v1, v2);
    case ATOMIC_SYSTEM:
      return atomicCAS_system(address, v1, v2);
  }
  // Should never happen
  return 0;
#endif
}

template <typename T, SCAMPAtomicType type>
__device__ inline T do_atomicAdd(T *address, T amount) {
#if __CUDA_ARCH__ < 600
  return ::atomicAdd(address, amount);
#else
  switch (type) {
    case ATOMIC_BLOCK:
      return atomicAdd_block(address, amount);
    case ATOMIC_GLOBAL:
      return atomicAdd(address, amount);
    case ATOMIC_SYSTEM:
      return atomicAdd_system(address, amount);
  }
  // Should never happen
  return 0;
#endif
}

template <typename T, SCAMPAtomicType type>
__device__ __forceinline__ T do_atomicMax(T *address, T other) {
#if __CUDA_ARCH__ < 600
  return atomicMax(address, other);
#else
  switch (type) {
    case ATOMIC_BLOCK:
      return atomicMax_block(address, other);
    case ATOMIC_GLOBAL:
      return atomicMax(address, other);
    case ATOMIC_SYSTEM:
      return atomicMax_system(address, other);
  }
  // Should never happen
  return 0;
#endif
}

template <typename T, SCAMPAtomicType type>
__device__ __forceinline__ T do_atomicMin(T *address, T other) {
#if __CUDA_ARCH__ < 600
  return atomicMax(address, other);
#else
  switch (type) {
    case ATOMIC_BLOCK:
      return atomicMin_block(address, other);
    case ATOMIC_GLOBAL:
      return atomicMin(address, other);
    case ATOMIC_SYSTEM:
      return atomicMin_system(address, other);
  }
  // Should never happen
  return 0;
#endif
}

template <SCAMPAtomicType type>
__device__ __forceinline__ float fAtomicMax(float *addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(do_atomicMax<int, type>(
                           (int *)addr, __float_as_int(value)))
                     : __uint_as_float(do_atomicMin<unsigned int, type>(
                           (unsigned int *)addr, __float_as_uint(value)));
  return old;
}

template <SCAMPAtomicType type>
__device__ __forceinline__ float fAtomicMax_check(float *addr, float value,
                                                  float check) {
  if (value < check) {
    return check;
  }
  return fAtomicMax<type>(addr, value);
}

// Atomically updates the MP/idxs using a single 64-bit integer. We lose a small
// amount of precision in the output, if we do not do this we are unable
// to atomically update both the matrix profile and the indexes without using a
// critical section and dedicated locks.
template <SCAMPAtomicType type>
__device__ inline void MPatomicMax(volatile uint64_t *address, float val,
                                   unsigned int idx) {
  mp_entry loc, loctest;
  loc.floats[0] = val;
  loc.ints[1] = idx;
  loctest.ulong = *address;
  while (loctest.floats[0] < val) {
    loctest.ulong = do_atomicCAS<unsigned long long int, type>(
        (unsigned long long int *)address, loctest.ulong, loc.ulong);
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
      loctest.ulong = do_atomicCAS<unsigned long long int, ATOMIC_BLOCK>(
          (unsigned long long int *)address, loctest.ulong, loc.ulong);
    }
  }
}

// Gets the max of 4 values (avoids returning NaN if any of d1-d4 are NaN)
template <typename T>
__device__ inline T max4(const T &d1, const T &d2, const T &d3, const T &d4) {
  float ret = -2;
  if (d1 > ret) {
    ret = d1;
  }
  if (d2 > ret) {
    ret = d2;
  }
  if (d3 > ret) {
    ret = d3;
  }
  if (d4 > ret) {
    ret = d4;
  }
  return ret;
}

// Gets the max of 4 values (avoids returning NaN if any of d1-d4 are NaN)
// Including the index
template <typename T>
__device__ inline T max4_index(const T &d1, const T &d2, const T &d3,
                               const T &d4, const uint32_t init,
                               uint32_t &idx) {
  float ret = -2;
  if (d1 > ret) {
    ret = d1;
    idx = init;
  }
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

/////////////////////////////////////////////////////////////////////////////////////
//     THESE HEADERS DEFINE VARIOUS COMPUTE STRATEGIES USED TO COMPUTE VARIOUS
//     PROFILE TYPES
///////////////////////////////////////////////////////////////////////////////////

#include "kernels_compute.h"
#include "kernels_smem.h"

/////////////////////////////////////////////////////////////////////////////////////
//
//  SCAMP TACTIC DESCRIBES STRATEGY FOR WHAT OPS TO EXECUTE IN THE KERNEL
//
/////////////////////////////////////////////////////////////////////////////////////

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class SCAMPTactic {
 public:
  __device__ SCAMPTactic() {}
  __device__ void InitMem(
      SCAMPKernelInputArgs<double> &args,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem,
      PROFILE_DATA_TYPE *__restrict__ profile_a,
      PROFILE_DATA_TYPE *__restrict__ profile_b, uint32_t col_start,
      uint32_t row_start) {
    _init_mem.exec(args, smem, profile_a, profile_b, col_start, row_start);
  }
  __device__ inline FORCE_INLINE void DoIteration(
      SCAMPThreadInfo<ACCUM_TYPE> &info,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem,
      OptionalArgs &args) {
    _do_iteration.exec(info, smem, args);
  }
  __device__ inline void WriteBack(SCAMPKernelInputArgs<double> &args,
                                   uint32_t tile_start_x, uint32_t tile_start_y,
                                   uint32_t n_x, uint32_t n_y,
                                   PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                                   PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                                   PROFILE_DATA_TYPE *__restrict__ profile_A,
                                   PROFILE_DATA_TYPE *__restrict__ profile_B) {
    _do_writeback.exec(args, tile_start_x, tile_start_y, n_x, n_y, local_mp_col,
                       local_mp_row, profile_A, profile_B);
  }

 private:
  InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                  TILE_WIDTH, TILE_HEIGHT, BLOCKSZ, PROFILE_TYPE>
      _init_mem;
  DoIterationStrategy<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                      PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                      COMPUTE_COLS, PROFILE_TYPE>
      _do_iteration;
  WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS, TILE_WIDTH,
                    TILE_HEIGHT, BLOCKSZ, PROFILE_TYPE>
      _do_writeback;
};

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
  constexpr int tile_width = tile_height + BLOCKSZ * DIAGS_PER_THREAD;
  SCAMPTactic<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, PROFILE_DATA_TYPE,
              ACCUM_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, tile_width,
              tile_height, BLOCKSZ, PROFILE_TYPE>
      tactic;
  SCAMPThreadInfo<ACCUM_TYPE> thread_info;

  extern __shared__ char smem_raw[];

  // Wrap the shared memory in  a struct which contains handles shared memory
  // accesses
  SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> smem(
      smem_raw, COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height,
      args.opt.num_extra_operands);

  // Find the starting diagonal of the distance matrix
  const unsigned int start_diag = (threadIdx.x * DIAGS_PER_THREAD) +
                                  blockIdx.x * (blockDim.x * DIAGS_PER_THREAD);

  // This is the index of the meta-diagonal that this thread block will work on
  const unsigned int meta_diagonal_idx = blockIdx.x;

  // The first diagonals constitiure a trivial match between the same
  // subsequence, we must exclude these from the calculation according to
  // args.exclusion_lower
  uint32_t tile_start_col =
      meta_diagonal_idx * (BLOCKSZ * DIAGS_PER_THREAD) + args.exclusion_lower;
  uint32_t tile_start_row = 0;

  // Initialize the column and row position of the current thread
  thread_info.global_col = tile_start_col + threadIdx.x * DIAGS_PER_THREAD;
  thread_info.global_row = 0;

  // num_diags is the number of diagonals in the distance matrix, less any
  // diagonals at the end we are not computing
  const unsigned int num_diags = args.n_x - args.exclusion_upper + 1;

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
  // cut from a diagonal slice of the distance matrix. Each thread starts on the
  // first row and works its way down-right towards right side of the distance
  // matrix
  while (tile_start_col < args.n_x && tile_start_row < args.n_y) {
    // Initialize the next tile's shared memory
    tactic.InitMem(args, smem, profile_A, profile_B, tile_start_col,
                   tile_start_row);
    thread_info.local_col = threadIdx.x * DIAGS_PER_THREAD;
    thread_info.local_row = 0;

    // Start of new tile, sync so we don't have data races with shared memory
    // initializaton
    __syncthreads();

    // There are 2 pathways here, most of the time we take the fast path (top),
    // the last tile in every thread-block will take the slower path (bottom)
    if (tile_start_col + tile_width < args.n_x &&
        tile_start_row + tile_height < args.n_y &&
        start_diag + DIAGS_PER_THREAD - 1 < num_diags) {
      // Fast Path
      while (thread_info.local_row < tile_height) {
        tactic.DoIteration(thread_info, smem, args.opt);
      }

    } else if (start_diag < num_diags) {
      // Slow Path
      while (thread_info.global_col < args.n_x &&
             thread_info.global_row < args.n_y &&
             thread_info.local_row < tile_height) {
        do_row_edge<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                    PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS>(
            thread_info, smem, args.n_x, start_diag, num_diags, args.opt);
        ++thread_info.global_col;
        ++thread_info.global_row;
        ++thread_info.local_col;
        ++thread_info.local_row;
      }
    }

    // After this sync, the caches will be updated with the best so far values
    // for this tile
    __syncthreads();

    // Write back our best-so-far computed for this tile to global memory
    tactic.WriteBack(args, tile_start_col, tile_start_row, args.n_x, args.n_y,
                     smem.local_mp_col, smem.local_mp_row, profile_A,
                     profile_B);

    // Update the tile position
    tile_start_col += tile_height;
    tile_start_row += tile_height;

    // Make sure our updates were committed before we pull in the next tile
    __threadfence_block();
  }
}

int get_blocksz(SCAMPPrecisionType t, const cudaDeviceProp &dev_prop) {
  if (t == PRECISION_DOUBLE) {
    return BLOCKSZ_DP;
  } else {
    return BLOCKSZ_SP;
  }
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

int get_smem(const OpInfo *info, uint64_t blocksz) {
  constexpr int num_shared_variables = 3;
  int intermediate_data_size = FPTypeSize(info->fp_type);
  int tile_height = GetTileHeight(info->fp_type);
  int tile_width = blocksz * DIAGS_PER_THREAD + tile_height;
  int smem = (tile_width + tile_height) *
             (num_shared_variables + info->opt_args.num_extra_operands) *
             intermediate_data_size;
  int profile_data_size = GetProfileTypeSize(info->profile_type);
  if (info->computing_cols) {
    smem += tile_width * profile_data_size;
  }
  if (info->computing_rows) {
    smem += tile_height * profile_data_size;
  }
  return smem;
}

template <typename PROFILE_DATA_TYPE, typename DISTANCE_TYPE,
          SCAMPProfileType PROFILE_TYPE, int BLOCKSPERSM>
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
        do_tile<double, double2, double4, double, PROFILE_DATA_TYPE,
                DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE,
                BLOCKSPERSM, TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, DISTANCE_TYPE,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, DISTANCE_TYPE,
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
        do_tile<double, double2, double4, double, PROFILE_DATA_TYPE,
                DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE,
                BLOCKSPERSM, TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, DISTANCE_TYPE,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, DISTANCE_TYPE,
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
        do_tile<double, double2, double4, double, PROFILE_DATA_TYPE,
                DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE,
                BLOCKSPERSM, TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_DATA_TYPE, DISTANCE_TYPE,
                COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE, BLOCKSPERSM,
                TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_DATA_TYPE, DISTANCE_TYPE,
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

SCAMPError_t compute_gpu_resources_and_launch(SCAMPKernelInputArgs<double> args,
                                              Tile *t, void *profile_a,
                                              void *profile_b, bool do_rows,
                                              bool do_cols) {
  int exclusion_total = args.exclusion_lower + args.exclusion_upper;
  uint64_t blocksz = get_blocksz(t->info()->fp_type, t->get_dev_props());
  uint64_t num_workers = ceil((args.n_x - exclusion_total) /
                              static_cast<double>(DIAGS_PER_THREAD));
  uint64_t num_blocks = ceil(num_workers / static_cast<double>(blocksz));
  uint64_t smem = get_smem(t->info(), blocksz);
  if (exclusion_total < args.n_x) {
    switch (t->info()->profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, double, PROFILE_TYPE_SUM_THRESH,
                            BLOCKSPERSM>(
            args, reinterpret_cast<double *>(profile_a),
            reinterpret_cast<double *>(profile_b), t->info()->fp_type, do_rows,
            do_cols, blocksz, num_blocks, smem, t->get_stream());
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, float, PROFILE_TYPE_1NN_INDEX,
                            BLOCKSPERSM>(
            args, reinterpret_cast<uint64_t *>(profile_a),
            reinterpret_cast<uint64_t *>(profile_b), t->info()->fp_type,
            do_rows, do_cols, blocksz, num_blocks, smem, t->get_stream());
      case PROFILE_TYPE_1NN:
        return LaunchDoTile<float, float, PROFILE_TYPE_1NN, BLOCKSPERSM>(
            args, reinterpret_cast<float *>(profile_a),
            reinterpret_cast<float *>(profile_b), t->info()->fp_type, do_rows,
            do_cols, blocksz, num_blocks, smem, t->get_stream());
      case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
        return LaunchDoTile<uint64_t, float, PROFILE_TYPE_APPROX_ALL_NEIGHBORS,
                            BLOCKSPERSM>(
            args, reinterpret_cast<uint64_t *>(profile_a),
            reinterpret_cast<uint64_t *>(profile_b), t->info()->fp_type,
            do_rows, do_cols, blocksz, num_blocks, smem, t->get_stream());
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t gpu_kernel_self_join_upper(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, false, false);
  if (t->info()->profile_type == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    return compute_gpu_resources_and_launch(tile_args, t, t->profile_a(),
                                            nullptr, t->info()->computing_rows,
                                            t->info()->computing_cols);
  }
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t gpu_kernel_self_join_lower(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, true, false);
  if (t->info()->profile_type == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    return compute_gpu_resources_and_launch(tile_args, t, t->profile_a(),
                                            nullptr, t->info()->computing_rows,
                                            t->info()->computing_cols);
  }
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}

SCAMPError_t gpu_kernel_ab_join_upper(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, false, true);
  if (t->info()->profile_type == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    return compute_gpu_resources_and_launch(tile_args, t, t->profile_a(),
                                            nullptr, t->info()->computing_rows,
                                            t->info()->computing_cols);
  }
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t gpu_kernel_ab_join_lower(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, true, true);
  if (t->info()->profile_type == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    return compute_gpu_resources_and_launch(tile_args, t, t->profile_a(),
                                            nullptr, t->info()->computing_rows,
                                            t->info()->computing_cols);
  }
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}
}  // namespace SCAMP
