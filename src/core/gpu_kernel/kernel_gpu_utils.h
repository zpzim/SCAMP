#pragma once
#include "common/common.h"

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

// Number of diagonals computed per thread
constexpr int DIAGS_PER_THREAD = 2;
constexpr int unrolled_diags = 2;
// Number of rows unrolled in the inner loop.
constexpr int unrolled_rows = 2;
constexpr int outer_unrolled_rows = 16;
constexpr int inner_unrolled_cols = unrolled_diags + unrolled_rows - 1;
// Number of columns unrolled in the inner loop.
constexpr int unrolled_cols = DIAGS_PER_THREAD + outer_unrolled_rows - 1;

// Number of iterations of the inner loop to do before syncing.
constexpr int KERNEL_TILE_ITERS = 16;
// Height of the parallelogram computed in the inner loop.
constexpr int KERNEL_TILE_HEIGHT = KERNEL_TILE_ITERS * outer_unrolled_rows;
constexpr int TILE_HEIGHT_SP = KERNEL_TILE_HEIGHT;
constexpr int TILE_HEIGHT_DP = KERNEL_TILE_HEIGHT;

// Threads per block for single precision SCAMP
constexpr int BLOCKSZ_SP = 512;
// Threads per block for double precision SCAMP
constexpr int BLOCKSZ_DP = 256;
// Minimum Blocks per SM for SCAMP
constexpr int BLOCKSPERSM = 2;

// Describes the SCOPE of an atomic operation in a GPU kernel
enum SCAMPAtomicType { ATOMIC_BLOCK, ATOMIC_GLOBAL, ATOMIC_SYSTEM };

HOST_DEVICE_FUNCTION constexpr bool NeedsCheckIfDone(
    SCAMPProfileType profile_type) {
  return profile_type == PROFILE_TYPE_APPROX_ALL_NEIGHBORS;
}

// Gets the profile element size as used by the GPU kernels
// This can be different than what is used in the CPU case
size_t GetProfileTypeSizeInternalGPU(SCAMPProfileType type);

// Get the desired block size to launch the kernel with according to tils
int get_blocksz(SCAMPPrecisionType fp_type);

// Gets the required amount of shared memory for the kernel
int get_smem(const OpInfo *info, uint64_t blocksz);

// Gets the tile height used by the kernel
int GetTileHeight(SCAMPPrecisionType dtype);

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
  if (value > check) {
    return fAtomicMax<type>(addr, value);
  }
  return -2;
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
