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

#if __CUDA_ARCH__ < 600
// Double atomicAdd is not implemented in hardware before Pascal, providing a
// software implementation here
__device__ double atomicAdd(double *address, double val) {
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

template <SCAMPAtomicType type>
__device__ inline unsigned long long do_atomicCAS(unsigned long long *address,
                                                  unsigned long long v1,
                                                  unsigned long long v2) {
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
__device__ inline uint32_t do_atomicAdd(T *address, T amount) {
#if __CUDA_ARCH__ < 600
  return atomicAdd(address, amount);
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
    loctest.ulong = do_atomicCAS<type>((unsigned long long int *)address,
                                       loctest.ulong, loc.ulong);
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
      loctest.ulong = do_atomicCAS<ATOMIC_BLOCK>(
          (unsigned long long int *)address, loctest.ulong, loc.ulong);
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
  __device__ void InitMem(SCAMPKernelInputArgs<double> &args,
                          SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                          PROFILE_DATA_TYPE *__restrict__ profile_a,
                          PROFILE_DATA_TYPE *__restrict__ profile_b,
                          uint32_t col_start, uint32_t row_start) {
    _init_mem.exec(args, smem, profile_a, profile_b, col_start, row_start);
  }
  __device__ inline __attribute__((always_inline)) void DoIteration(
      SCAMPThreadInfo<ACCUM_TYPE> &info,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem, OptionalArgs &args) {
    _do_iteration.exec(info, smem, args);
  }
  __device__ inline void DoEdge(int i, int j, int x, int y, int n,
                                ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                                ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                                size_t num_diags,
                                SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                                OptionalArgs &args) {
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

 private:
  InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                  TILE_WIDTH, TILE_HEIGHT, BLOCKSZ, PROFILE_TYPE>
      _init_mem;
  DoIterationStrategy<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                      PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                      COMPUTE_COLS, PROFILE_TYPE>
      _do_iteration;
  DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                    COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>
      _do_edge;
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
  constexpr int diags_per_thread = 4;
  constexpr int tile_width = tile_height + BLOCKSZ * diags_per_thread;
  SCAMPTactic<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, PROFILE_DATA_TYPE,
              ACCUM_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, tile_width,
              tile_height, BLOCKSZ, PROFILE_TYPE>
      tactic;
  SCAMPThreadInfo<ACCUM_TYPE> thread_info;

  extern __shared__ char smem_raw[];
  SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> smem(
      smem_raw, COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height);

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
                     smem.local_mp_col, smem.local_mp_row, profile_A,
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
  if (start_column >= start_row && start_column <= start_row + exclusion) {
    return exclusion;
  }
  return 0;
}

std::pair<int, int> get_exclusion_for_ab_join(uint64_t window_size,
                                              uint64_t start_row,
                                              uint64_t start_column,
                                              bool upper_tile, int tile_dim) {
  int exclusion_lower = 0;
  int exclusion_upper = 0;
  if (upper_tile) {
    exclusion_lower = get_exclusion(window_size, start_row, start_column);
    if (start_row > start_column) {
      exclusion_upper =
          get_exclusion(window_size, start_row, start_column + tile_dim);
    } else {
      exclusion_upper = 0;
    }
    return std::make_pair(exclusion_lower, exclusion_upper);
  }
  exclusion_lower = get_exclusion(window_size, start_column, start_row);
  if (start_row >= start_column) {
    exclusion_upper = 0;
  } else {
    exclusion_upper =
        get_exclusion(window_size, start_column, start_row + tile_dim);
  }
  return std::make_pair(exclusion_lower, exclusion_upper);
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
  if (computing_cols) {
    smem += tile_width * profile_data_size;
  }
  if (computing_rows) {
    smem += tile_height * profile_data_size;
  }
  return smem;
}

template <typename PROFILE_DATA_TYPE, SCAMPProfileType PROFILE_TYPE,
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
  } else if (computing_cols) {
    constexpr bool COMPUTE_COLS = true;
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
  } else if (computing_rows) {
    constexpr bool COMPUTE_COLS = false;
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
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t gpu_kernel_self_join_upper(
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
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, true, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_SELF>(
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

SCAMPError_t gpu_kernel_self_join_lower(
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
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, true, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_SELF>(
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
SCAMPError_t gpu_kernel_ab_join_upper(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    int64_t distributed_col, int64_t distributed_row, bool aligned_ab_join,
    const cudaDeviceProp &props, SCAMPPrecisionType t, bool computing_rows,
    const OptionalArgs &args, SCAMPProfileType profile_type, cudaStream_t s) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  std::pair<int, int> exclusion_pair(0, 0);
  if (aligned_ab_join) {
    int start_col = global_col;
    int start_row = global_row;
    if (distributed_col >= 0 && distributed_row >= 0) {
      start_col += distributed_col;
      start_row += distributed_row;
    }
    exclusion_pair = get_exclusion_for_ab_join(window_size, start_row,
                                               start_col, true, tile_width);
  }
  uint64_t num_workers =
      ceil((tile_width - (exclusion_pair.first + exclusion_pair.second)) /
           (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(
      QT, df_A, df_B, dg_A, dg_B, norms_A, norms_B, tile_width, tile_height,
      exclusion_pair.first, exclusion_pair.second, args);
  uint64_t smem = get_smem(computing_rows, true, blocksz, t,
                           GetProfileTypeSize(profile_type));
  if ((exclusion_pair.first + exclusion_pair.second) < tile_width) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, computing_rows, true, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_AB>(
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

SCAMPError_t gpu_kernel_ab_join_lower(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    int64_t distributed_col, int64_t distributed_row, bool aligned_ab_join,
    const cudaDeviceProp &props, SCAMPPrecisionType t, bool computing_rows,
    const OptionalArgs &args, SCAMPProfileType profile_type, cudaStream_t s) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  std::pair<int, int> exclusion_pair;
  if (aligned_ab_join) {
    int start_col = global_col;
    int start_row = global_row;
    if (distributed_col >= 0 && distributed_row >= 0) {
      start_col += distributed_col;
      start_row += distributed_row;
    }
    exclusion_pair = get_exclusion_for_ab_join(window_size, start_row,
                                               start_col, false, tile_height);
  }
  uint64_t num_workers =
      ceil((tile_height - (exclusion_pair.first + exclusion_pair.second)) /
           (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(
      QT, df_B, df_A, dg_B, dg_A, norms_B, norms_A, tile_height, tile_width,
      exclusion_pair.first, exclusion_pair.second, args);
  uint64_t smem = get_smem(computing_rows, true, blocksz, t,
                           GetProfileTypeSize(profile_type));
  if (exclusion_pair.first + exclusion_pair.second < tile_height) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, computing_rows, blocksz, num_blocks, smem, s);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_AB>(
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
