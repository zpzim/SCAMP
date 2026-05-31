#pragma once
#include <Eigen/Core>
#include <utility>

#include "common/common.h"
#include "core/kernel_common.h"
#include "core/tile.h"

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

// Describes the SCOPE of an atomic operation in a GPU kernel
enum SCAMPAtomicType { ATOMIC_BLOCK, ATOMIC_GLOBAL, ATOMIC_SYSTEM };

// Hardware max threads/SM for the CUDA arch we're currently compiling for.
// Used as the upper cap on `__launch_bounds__(BLOCKSZ, bps)` so that the
// implied BLOCKSZ*bps never exceeds the per-SM thread ceiling (which would
// otherwise be silently clamped by ptxas, leaving the actual occupancy
// nondeterministic).
//
// Values from the CUDA C++ Programming Guide "Compute Capability" tables.
// During host compilation (__CUDA_ARCH__ undefined) we return the more
// permissive 2048 so any host-side use of this constant doesn't accidentally
// under-cap; the launch_bounds itself is only consumed by ptxas during
// device compilation where the right __CUDA_ARCH__ guard applies.
constexpr int hw_threads_per_sm() {
#if defined(__CUDA_ARCH__)
  // Turing has half the warps per SM compared to neighbouring arches.
  if constexpr (__CUDA_ARCH__ == 750) return 1024;
  // GA10x, Orin, Ada GeForce, Jetson Thor, Blackwell GeForce: 1536 threads/SM.
  if constexpr (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 ||
                __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1100 ||
                __CUDA_ARCH__ == 1200) {
    return 1536;
  }
#endif
  // Maxwell, Pascal, Volta, A100, Hopper, Blackwell datacenter, and host.
  return 2048;
}

// Compute a launch_bounds-friendly blocks_per_sm value from the variant's
// target thread density (typically `blocks_per_sm * default_blocksz` from
// the variant tuple) and the actual BLOCKSZ this template was instantiated
// for. The target stays constant across the autotuner's blocksz sweep so
// the per-thread register budget stays uniform; the hw cap (defaulting to
// the per-arch hw_threads_per_sm()) prevents ptxas from silently clamping
// when BLOCKSZ * variant_bps exceeds the per-SM thread ceiling.
//
// Returns max(1, min(target/blocksz, hw/blocksz)).
//
// The hw_threads default arg is evaluated at the call site, so when the
// usual __launch_bounds__(BLOCKSZ, safe_bps(target, BLOCKSZ)) call expands
// in __global__ code, hw_threads_per_sm() resolves under the current
// __CUDA_ARCH__ guard and the per-arch hw cap is baked into each per-arch
// device compilation pass. Callers that explicitly want a different cap
// (e.g. host-side overrides for testing) pass it as the third argument.
constexpr int safe_bps(int target_threads_per_sm, int blocksz,
                       int hw_threads = hw_threads_per_sm()) {
  if (blocksz <= 0) return 1;
  const int desired = target_threads_per_sm / blocksz;
  const int hw_cap = hw_threads / blocksz;
  const int result = desired < hw_cap ? desired : hw_cap;
  return result > 0 ? result : 1;
}

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

  // MATRIX_SUMMARY per-cell coalescing grid. Aliases the local_mp_col smem
  // region (matrix summary has no per-column profile, so that region is free).
  // The block accumulates per-cell maxes here with block-scoped atomics and
  // flushes to the global matrix once per stripe-step; this turns thousands of
  // scattered global atomicMax into a handful. Bounds (ms_*_min / ms_grid_*)
  // are recomputed per stripe-step in init_smem; ms_matrix is the global output
  // matrix used for the flush and the out-of-grid fallback.
  float *ms_grid;
  float *ms_matrix;
  int ms_col_min, ms_row_min, ms_grid_w, ms_grid_h, ms_num_cells;
  bool ms_use_grid;
  bool ms_rowwise;
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
  // The matrix-summary cell grid reuses whichever per-profile region this run
  // allocated but does not otherwise use: local_mp_col on the normal
  // (compute_columns) run, local_mp_row on the transposed (compute_rows) run.
  // init_smem sizes/initializes it per stripe-step. Only consumed when
  // PROFILE_TYPE == MATRIX_SUMMARY.
  ms_grid = compute_columns
                ? reinterpret_cast<float *>(local_mp_col.data())
                : (compute_rows ? reinterpret_cast<float *>(local_mp_row.data())
                                : nullptr);
  ms_matrix = nullptr;
  ms_col_min = ms_row_min = 0;
  ms_grid_w = ms_grid_h = ms_num_cells = 0;
  ms_use_grid = false;
  ms_rowwise = false;
}

template <typename ACCUM_TYPE, int DiagsPerThread>
struct SCAMPThreadInfo {
  Eigen::Array<ACCUM_TYPE, DiagsPerThread, 1> cov;
  uint32_t local_row;
  uint32_t local_col;
  uint32_t global_row;
  uint32_t global_col;
  // MATRIX_SUMMARY register accumulator: running max for the current output
  // cell. ms_cell is the linearized global cell index, -1 when empty. Flushed
  // to the smem grid on a cell change and once at stripe end before write_back.
  int ms_cell;
  float ms_max;
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

// vec_load<N, AlignBytes, T>: load N elements of type T from a shared-memory
// pointer with compile-time-known alignment, writing them into dst. Picks the
// widest aligned PTX vector load supported by the (T, AlignBytes) tuple and
// recursively peels off the remaining tail.
//
// Why this exists: Eigen::Map<Array<T,N,1>>::segment<>(off) assignments
// compile to N scalar `ld.shared.{f32,f64}` instructions under nvcc — its
// packet path is mostly inactive in device code. The pre-Eigen master used
// raw `reinterpret_cast<float4 *>(smem.df_col)[c]` to issue
// `ld.shared.v4.f32` (one transaction, 4 elements). We reclaim that
// optimization here without giving up the Eigen Array body of the kernel.
//
// AlignBytes is a *worst-case* alignment guarantee from the caller, NOT a
// runtime check. Passing too generous a value is UB. For example, in the
// SCAMP kernel:
//   - smem.df_col.data() + info.local_col is aligned to
//     DiagsPerThread * sizeof(T) bytes (local_col = threadIdx.x *
//     DiagsPerThread).
//   - smem.df_row.data() + info.local_row + j*UR is aligned to
//     UnrolledRows * sizeof(T) bytes (local_row is a multiple of
//     OuterUnrolledRows, j*UR is a multiple of UR, OUR is a multiple of UR).
//   - The sliding-window refill in do_iteration_fast lands at offset
//     local_col + j*UR + (DiagsPerThread - 1), which is NOT cleanly aligned
//     for any vector type — that load stays scalar.
//
// nvcc shared-memory PTX vector load support (sm_8x):
//   - 32-bit elements: ld.shared.v4.f32 (16B), ld.shared.v2.f32 (8B)
//   - 64-bit elements: ld.shared.v2.f64 (16B); no v4 for 64-bit on smem
template <int N, int AlignBytes, typename T>
__device__ inline void vec_load(const T *p, T *d) {
  static_assert(N >= 0, "vec_load N must be non-negative.");
  if constexpr (N == 0) {
    return;
  } else if constexpr (sizeof(T) == 4 && N >= 4 && AlignBytes >= 16) {
    float4 v = *reinterpret_cast<const float4 *>(p);
    d[0] = v.x;
    d[1] = v.y;
    d[2] = v.z;
    d[3] = v.w;
    // After loading 4 floats from a 16B-aligned addr, the next addr is also
    // 16B-aligned; alignment guarantee carries through the recursion.
    vec_load<N - 4, 16, T>(p + 4, d + 4);
  } else if constexpr (sizeof(T) == 4 && N >= 2 && AlignBytes >= 8) {
    float2 v = *reinterpret_cast<const float2 *>(p);
    d[0] = v.x;
    d[1] = v.y;
    vec_load<N - 2, 8, T>(p + 2, d + 2);
  } else if constexpr (sizeof(T) == 8 && N >= 2 && AlignBytes >= 16) {
    double2 v = *reinterpret_cast<const double2 *>(p);
    d[0] = v.x;
    d[1] = v.y;
    vec_load<N - 2, 16, T>(p + 2, d + 2);
  } else {
    // Fall through: scalar load, then recurse with a conservatively reduced
    // alignment (we just consumed sizeof(T) bytes from a possibly larger
    // aligned region; only the natural alignment of the scalar survives).
    d[0] = p[0];
    vec_load<N - 1, sizeof(T), T>(p + 1, d + 1);
  }
}

// Gets the profile element size as used by the GPU kernels
// This can be different than what is used in the CPU case
size_t GetProfileTypeSizeInternalGPU(SCAMPProfileType type);

// Gets the required amount of shared memory for the kernel. tile_height and
// diags_per_thread are per-variant; passing them explicitly lets the
// autotuner size smem for the variant it picked.
int get_smem(const OpInfo *info, uint64_t blocksz, int tile_height,
             int diags_per_thread);

// Same but for the cov-shuffle "shfl" variant: skips the column data smem
// regions (df_col/dg_col/inorm_col live in per-lane registers in the shfl
// kernel) and adds the cov_handoff hand-off region (2 * warps_per_block
// scalars, double-buffered).
int get_smem_shfl(const OpInfo *info, uint64_t blocksz, int tile_height,
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

// ----------------------------------------------------------------------------
// MATRIX_SUMMARY per-cell reduction (mirrors the CPU update_mp bucketing).
//
// MATRIX_SUMMARY downsamples the full distance matrix into a matrix_height x
// matrix_width grid where cell (R, C) = max correlation over every (row, col)
// of the distance matrix that maps into that cell. The CPU does this per-cell
// (cpu_kernels.cpp update_mp); we mirror that exactly here so the GPU result
// matches the CPU reference. To keep it fast we don't atomicMax into the small
// global grid per cell (uncoalesced, latency-bound) -- we accumulate into a
// per-block smem grid covering only the cells this stripe-step touches, then
// flush once. ms_accumulate_cell routes a single distance into either the smem
// grid (common case) or, if its cell falls outside the precomputed grid window
// or the grid was disabled, straight to the global matrix (never dropped).
// ----------------------------------------------------------------------------

// Output-matrix (row_bucket, col_bucket) for a tile-local distance-matrix cell
// at integer position (gr, gc). The bucket boundary math (cols_per_cell /
// rows_per_cell) is done in float, not double: on consumer GPUs FP64 has ~1/64
// the throughput of FP32, and this per-cell division otherwise dominates the
// matrix-summary kernel. Positions are < 2^24 for all non-distributed inputs
// so the float cast is exact; only the rare exact-boundary column can land in a
// different cell than the double reference (absorbed by the test tolerance).
// `rowwise` mirrors the CPU's transposed branch (lower/transposed kernel run):
// the local row drives the column bucket and vice versa.
__device__ inline void ms_cell_of(int gr, int gc, bool rowwise,
                                  const SCAMPKernelInputArgs<double> &args,
                                  int *row_b, int *col_b) {
  int for_col = rowwise ? gr : gc;
  int for_row = rowwise ? gc : gr;
  // Float division: off the FP64 pipe, but more accurate at cell boundaries
  // than a reciprocal multiply (no separate 1/c rounding step). cols/rows_per
  // _cell are already float in the kernel args.
  *col_b = static_cast<int>(floorf((static_cast<float>(for_col) +
                                    static_cast<float>(args.global_start_col)) /
                                   args.cols_per_cell));
  *row_b = static_cast<int>(floorf((static_cast<float>(for_row) +
                                    static_cast<float>(args.global_start_row)) /
                                   args.rows_per_cell));
}

// Write one (row_b, col_b, corr) into the smem cell grid with a block-scoped
// atomicMax, or -- if the cell falls outside the precomputed grid window or the
// grid is disabled -- straight to the global matrix. Never drops a value.
template <typename SMEM_T>
__device__ inline void ms_store_cell(SMEM_T &smem, int row_b, int col_b,
                                     float corr,
                                     const SCAMPKernelInputArgs<double> &args) {
  if (smem.ms_use_grid) {
    int lr = row_b - smem.ms_row_min;
    int lc = col_b - smem.ms_col_min;
    if (lr >= 0 && lr < smem.ms_grid_h && lc >= 0 && lc < smem.ms_grid_w) {
      fAtomicMax<ATOMIC_BLOCK>(smem.ms_grid + lr * smem.ms_grid_w + lc, corr);
      return;
    }
  }
  fAtomicMax<ATOMIC_GLOBAL>(
      smem.ms_matrix + static_cast<int64_t>(row_b) * args.matrix_width + col_b,
      corr);
}

// Flush a thread's pending register accumulator to the grid/matrix and empty
// it. info.ms_cell is the linearized global cell index, -1 when empty.
template <typename INFO_T, typename SMEM_T>
__device__ inline void ms_flush_accumulator(
    INFO_T &info, SMEM_T &smem, const SCAMPKernelInputArgs<double> &args) {
  if (info.ms_cell >= 0) {
    int row_b = info.ms_cell / args.matrix_width;
    int col_b = info.ms_cell - row_b * args.matrix_width;
    ms_store_cell(smem, row_b, col_b, info.ms_max, args);
    info.ms_cell = -1;
  }
}

// Read the current value of cell (row_b, col_b) from the smem grid or the
// global matrix fallback. Mirrors ms_store_cell's routing. Used at cell entry
// to seed the per-thread running max with whatever other threads / earlier
// stripe-steps have already deposited -- so a lane joining a cell that's
// already had a high value contributed doesn't bother re-issuing atomicMax for
// every position below that floor.
template <typename SMEM_T>
__device__ inline float ms_read_cell(SMEM_T &smem, int row_b, int col_b,
                                     const SCAMPKernelInputArgs<double> &args) {
  if (smem.ms_use_grid) {
    int lr = row_b - smem.ms_row_min;
    int lc = col_b - smem.ms_col_min;
    if (lr >= 0 && lr < smem.ms_grid_h && lc >= 0 && lc < smem.ms_grid_w) {
      return smem.ms_grid[lr * smem.ms_grid_w + lc];
    }
  }
  return smem
      .ms_matrix[static_cast<int64_t>(row_b) * args.matrix_width + col_b];
}

// Register-coalesced per-cell accumulate. Holds the running max for the current
// output cell in registers and only emits an atomic when the cell changes;
// consecutive distances almost always map to the same cell (cells span many
// rows/cols), so this collapses many atomics into one. The caller must flush
// the trailing accumulator (ms_flush_accumulator) before write_back reads the
// grid. Matches CPU update_mp: only contributions >= threshold count (NaN
// compares false and is skipped).
template <typename INFO_T, typename SMEM_T>
__device__ inline void ms_accumulate_cell(
    INFO_T &info, SMEM_T &smem, int gr, int gc, float corr, float thresh,
    const SCAMPKernelInputArgs<double> &args) {
  if (!(corr >= thresh)) {
    return;
  }
  int row_b, col_b;
  ms_cell_of(gr, gc, smem.ms_rowwise, args, &row_b, &col_b);
  int idx = row_b * args.matrix_width + col_b;
  if (idx == info.ms_cell) {
    info.ms_max = fmaxf(info.ms_max, corr);
  } else {
    ms_flush_accumulator(info, smem, args);
    info.ms_cell = idx;
    info.ms_max = fmaxf(ms_read_cell(smem, row_b, col_b, args), corr);
  }
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
