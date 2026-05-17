#pragma once
#include <Eigen/Core>
#include <utility>

#include "common/common.h"
#include "core/tile.h"
#include "kernel_constants.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Double atomicAdd is implemented
#else
// Double atomicAdd is not implemented before Pascal, providing a
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

// Launch constants moved to kernel_constants.h so they can be consumed from
// host-only translation units (e.g. kernel_config.cpp) without pulling in
// the CUDA device intrinsics defined later in this header.

// Describes the SCOPE of an atomic operation in a GPU kernel
enum SCAMPAtomicType { ATOMIC_BLOCK, ATOMIC_GLOBAL, ATOMIC_SYSTEM };

HOST_DEVICE_FUNCTION constexpr bool NeedsCheckIfDone(
    SCAMPProfileType profile_type) {
  return profile_type == PROFILE_TYPE_APPROX_ALL_NEIGHBORS;
}

// Structure which manages shared memory on the GPU and automatically allocates
// appropriate segments in memory for variables used by the kernel.
//
// The per-region pointers are wrapped in Eigen::Map<Eigen::Array<..., N, 1>>
// so callers can write expressions like smem.df_col.segment<unrolled_diags>(
// info.local_col) instead of hand-unrolled raw-pointer loads.
//
// Eigen::Map has no default constructor that takes nothing, so the ctor
// uses placement new to overwrite nullptr-initialized Map members with maps
// pointing into the smem buffer. (Eigen 5 doesn't expose a public mutator
// for the underlying pointer; placement new remains the canonical pattern
// for re-seating a Map.)
template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, SCAMPProfileType type,
          int tile_width, int tile_height>
struct SCAMPSmem {
  __device__ SCAMPSmem(char *smem, bool compute_rows, bool compute_columns,
                       int extra_operands);

  // Public typedef so callees can spell the scalar type of the column /
  // row data segments without re-deriving the SCAMPSmem template args.
  // Needed because for PRECISION_MIXED the cov accumulator's scalar type
  // (ACCUM_TYPE=double) differs from the column-data scalar type
  // (DATA_TYPE=float); local register-window arrays must use DataType to
  // match the segment they're loaded from.
  using DataType = DATA_TYPE;

  Eigen::Map<Eigen::Array<DATA_TYPE, tile_width, 1>> df_col;
  Eigen::Map<Eigen::Array<DATA_TYPE, tile_width, 1>> dg_col;
  Eigen::Map<Eigen::Array<DATA_TYPE, tile_width, 1>> inorm_col;
  Eigen::Map<Eigen::Array<DATA_TYPE, tile_height, 1>> df_row;
  Eigen::Map<Eigen::Array<DATA_TYPE, tile_height, 1>> dg_row;
  Eigen::Map<Eigen::Array<DATA_TYPE, tile_height, 1>> inorm_row;
  Eigen::Map<Eigen::Array<PROFILE_DATA_TYPE, tile_width, 1>> local_mp_col;
  Eigen::Map<Eigen::Array<PROFILE_DATA_TYPE, tile_height, 1>> local_mp_row;

  uint64_t *profile_a_length;
  uint64_t *profile_b_length;
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, SCAMPProfileType type,
          int tile_width, int tile_height>
__device__ SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, type, tile_width,
                     tile_height>::SCAMPSmem(char *smem, bool compute_rows,
                                             bool compute_columns,
                                             int extra_operands)
    : df_col(nullptr),
      dg_col(nullptr),
      inorm_col(nullptr),
      df_row(nullptr),
      dg_row(nullptr),
      inorm_row(nullptr),
      local_mp_col(nullptr),
      local_mp_row(nullptr) {
  using WideArrayMap = decltype(df_col);
  using TallArrayMap = decltype(df_row);
  using ColProfileMap = decltype(local_mp_col);
  using RowProfileMap = decltype(local_mp_row);

  new (&df_col) WideArrayMap(reinterpret_cast<DATA_TYPE *>(smem));
  smem += sizeof(DATA_TYPE) * tile_width;
  new (&dg_col) WideArrayMap(reinterpret_cast<DATA_TYPE *>(smem));
  smem += sizeof(DATA_TYPE) * tile_width;
  new (&inorm_col) WideArrayMap(reinterpret_cast<DATA_TYPE *>(smem));
  smem += sizeof(DATA_TYPE) * tile_width;
  new (&df_row) TallArrayMap(reinterpret_cast<DATA_TYPE *>(smem));
  smem += sizeof(DATA_TYPE) * tile_height;
  new (&dg_row) TallArrayMap(reinterpret_cast<DATA_TYPE *>(smem));
  smem += sizeof(DATA_TYPE) * tile_height;
  new (&inorm_row) TallArrayMap(reinterpret_cast<DATA_TYPE *>(smem));
  smem += sizeof(DATA_TYPE) * tile_height;

  if (compute_columns) {
    new (&local_mp_col)
        ColProfileMap(reinterpret_cast<PROFILE_DATA_TYPE *>(smem));
    smem += sizeof(PROFILE_DATA_TYPE) * tile_width;
  }
  // local_mp_col, local_mp_row Maps were nullptr-constructed above; if their
  // computing-* flag is false they remain null (and the kernel will not read
  // them).
  if (compute_rows) {
    new (&local_mp_row)
        RowProfileMap(reinterpret_cast<PROFILE_DATA_TYPE *>(smem));
    smem += sizeof(PROFILE_DATA_TYPE) * tile_height;
  }
  if (NeedsCheckIfDone(type)) {
    profile_a_length = reinterpret_cast<uint64_t *>(smem);
    smem += sizeof(uint64_t);
    profile_b_length = reinterpret_cast<uint64_t *>(smem);
  } else {
    profile_a_length = nullptr;
    profile_b_length = nullptr;
  }
}

template <typename ACCUM_TYPE, int DiagsPerThread>
struct SCAMPThreadInfo {
  Eigen::Array<ACCUM_TYPE, DiagsPerThread, 1> cov;
  uint32_t local_row;
  uint32_t local_col;
  uint32_t global_row;
  uint32_t global_col;
};

// Compile-time-unrolled for loop driven by std::index_sequence. Each
// invocation of `func` receives a `num<I>` value whose `::value` is the
// constexpr loop index, so loop-iter-dependent template arguments are
// usable inside the body. Faster than #pragma unroll for inner-kernel use
// because the index is a true compile-time constant.
template <std::size_t N>
struct num {
  static constexpr auto value = N;
};

template <class F, std::size_t... Is>
__device__ inline void for_(F func, std::index_sequence<Is...>) {
  using expander = int[];
  (void)expander{0, ((void)func(num<Is>{}), 0)...};
}

template <std::size_t N, typename F>
__device__ inline void for_(F func) {
  for_(func, std::make_index_sequence<N>());
}

// Gets the profile element size as used by the GPU kernels
// This can be different than what is used in the CPU case
size_t GetProfileTypeSizeInternalGPU(SCAMPProfileType type);

// Get the desired block size to launch the kernel with according to tils
int get_blocksz(Tile *t);

// Gets the required amount of shared memory for the kernel. tile_height and
// diags_per_thread are per-variant; passing them explicitly lets the
// autotuner size smem for the variant it picked.
int get_smem(const OpInfo *info, uint64_t blocksz, int tile_height,
             int diags_per_thread);

// Gets the size of an element for particular SCAMP precision type
int FPTypeSize(SCAMPPrecisionType dtype);

// Gets the max of 4 values (avoids returning NaN if any of d1-d4 are NaN)
template <typename T>
__device__ inline T max4(const T &d1, const T &d2, const T &d3, const T &d4) {
  T ret = -2;
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
  T ret = -2;
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

/////////////////////////////////////////////
// Atomic OPs for CUDA kernels
/////////////////////////////////////////////

// Atomic Max selector based on Atomic type and CUDA Arch
template <typename T, SCAMPAtomicType type>
__device__ inline T do_atomicMax(T *address, T other) {
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

// Atomic Min selector based on Atomic type and CUDA Arch
template <typename T, SCAMPAtomicType type>
__device__ inline T do_atomicMin(T *address, T other) {
#if __CUDA_ARCH__ < 600
  return ::atomicMin(address, other);
#else
  switch (type) {
    case ATOMIC_BLOCK:
      return ::atomicMin_block(address, other);
    case ATOMIC_GLOBAL:
      return ::atomicMin(address, other);
    case ATOMIC_SYSTEM:
      return ::atomicMin_system(address, other);
  }
  // Should never happen
  return 0;
#endif
}

// Atomic CAS selector based on Atomic type and CUDA Arch
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

// Atomic Add selector based on Atomic type and CUDA Arch
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

// Atomically updates the MP/idxs using a single 64-bit integer. We lose a small
// amount of precision in the output, if we do not do this we are unable
// to atomically update both the matrix profile and the indexes without using a
// critical section and dedicated locks.
template <SCAMPAtomicType type>
__device__ inline void MPatomicMax(uint64_t *address, float val,
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
template <SCAMPAtomicType type>
__device__ inline void MPatomicMax_check(uint64_t *address, float val,
                                         unsigned int idx, float curr_val) {
  if (val > curr_val) {
    MPatomicMax<type>(address, val, idx);
  }
}

// Atomic Max For single floating point calculations
template <SCAMPAtomicType type>
__device__ inline float fAtomicMax(float *addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(do_atomicMax<int, type>(
                           (int *)addr, __float_as_int(value)))
                     : __uint_as_float(do_atomicMin<unsigned int, type>(
                           (unsigned int *)addr, __float_as_uint(value)));
  return old;
}

// Atomic Max For single precision floating point, but with a check
template <SCAMPAtomicType type>
__device__ inline float fAtomicMax_check(float *addr, float value,
                                         float check) {
  if (value < check) {
    return check;
  }
  return fAtomicMax<type>(addr, value);
}

// Outputs an 'initial' distance value based on the type of profile being
// computed
template <typename DISTANCE_TYPE, SCAMPProfileType type>
__device__ inline DISTANCE_TYPE init_dist() {
  switch (type) {
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
    case PROFILE_TYPE_1NN_INDEX:
    case PROFILE_TYPE_1NN:
    case PROFILE_TYPE_MATRIX_SUMMARY:
      // Smallest value possible is -1 so set to -2
      return static_cast<DISTANCE_TYPE>(-2);
    case PROFILE_TYPE_SUM_THRESH:
    case PROFILE_TYPE_FREQUENCY_THRESH:
    default:
      // We must set to 0 so we get an accurate sum
      return static_cast<DISTANCE_TYPE>(0);
  }
}

}  // namespace SCAMP
