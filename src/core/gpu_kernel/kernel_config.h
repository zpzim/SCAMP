#pragma once
#include <cstddef>
#include <string>
#include "common/common.h"

namespace SCAMP {

// KernelConfig holds the runtime-tunable launch parameters for the SCAMP
// do_tile<...> CUDA kernel. Every field is a "variant axis" the
// autotuner can vary independently.
//
// blocksz             threads per block. The kernel is instantiated for
//                     {64,128,256,512} per variant; the autotuner sweeps
//                     this axis and the on-disk cache can store any of
//                     them. The cold-start fallback uses the per-variant
//                     default_blocksz_{dp,sp} declared in
//                     SCAMP_VARIANT_TUPLES (see src/core/gpu_kernel/
//                     CMakeLists.txt).
// blocks_per_sm       __launch_bounds__ second arg (occupancy hint).
// diags_per_thread    distance-matrix diagonals processed per thread per
//                     fast-path iteration. Wider = fewer threads, more
//                     work per thread.
// unrolled_rows       row batch size in do_iteration_fast's inner loop.
// outer_unrolled_rows rows processed per do_iteration_fast call.
// kernel_tile_iters   do_iteration_fast calls per tile.
//
// tile_height is NOT a field -- it's derived (kernel_tile_iters *
// outer_unrolled_rows). The smem layout depends on tile_height, so get_smem
// takes a tile_height arg.
//
// Each (blocks_per_sm, diags_per_thread, unrolled_rows, outer_unrolled_rows,
// kernel_tile_iters) tuple must have a matching branch in the LaunchDoTile
// switch in kernels_impl.h, otherwise IsSupportedKernelConfig would accept
// it but the launch would fall back to the default.
struct KernelConfig {
  int blocksz;
  int blocks_per_sm;
  int diags_per_thread;
  int unrolled_rows;
  int outer_unrolled_rows;
  int kernel_tile_iters;

  // Derived; kept as an inline helper so callers don't have to recompute.
  int tile_height() const { return kernel_tile_iters * outer_unrolled_rows; }

  bool operator==(const KernelConfig &o) const {
    return blocksz == o.blocksz && blocks_per_sm == o.blocks_per_sm &&
           diags_per_thread == o.diags_per_thread &&
           unrolled_rows == o.unrolled_rows &&
           outer_unrolled_rows == o.outer_unrolled_rows &&
           kernel_tile_iters == o.kernel_tile_iters;
  }
};

// Cold-start fallback when no autotune cache entry exists for the device
// + (profile, precision) tuple. Returns the first shfl variant (the
// safest universal default; see GetDefaultKernelConfig impl for the
// rationale). The autotune cache always wins over this when present.
KernelConfig GetDefaultKernelConfig(SCAMPPrecisionType precision);

// Enumerated launch-geometry variants, generated from SCAMP_VARIANT_TUPLES
// in src/core/gpu_kernel/CMakeLists.txt. Each variant's full tuple must
// have a matching branch in the LaunchDoTile switch in kernels_impl.h.
//
// default_blocksz_{dp,sp} is the per-precision cold-start blocksz. The
// kernel is instantiated for every blocksz in {64,128,256,512}, so the
// autotune sweep and the cache can pick any of them at runtime; these
// fields only feed GetKernelConfigForVariant when no cache hit exists.
struct KernelVariantGeometry {
  int blocks_per_sm;
  int diags_per_thread;
  int unrolled_rows;
  int outer_unrolled_rows;
  int kernel_tile_iters;
  // Variant author declares only the SP cold-start blocksz. The DP value
  // is implicit (= default_blocksz_sp / 2): DP uses 2x the registers per
  // thread, and halving threads/SM keeps the per-thread register budget
  // stable across precisions. Use DefaultBlockszForPrecision() to derive
  // the precision-specific value.
  int default_blocksz_sp;
};

extern const std::size_t kNumKernelVariants;

const KernelVariantGeometry &GetKernelVariantGeometry(std::size_t i);

// Materialize a full KernelConfig (including precision-derived blocksz) for
// the variant at index `i`. Asserts that i < kNumKernelVariants.
KernelConfig GetKernelConfigForVariant(std::size_t i,
                                       SCAMPPrecisionType precision);

// True iff the cfg matches any enumerated variant for `precision`. Used by
// GetKernelConfigForDevice to reject cache entries that name a config no
// current binary supports (older or newer build with a different variant
// set).
bool IsSupportedKernelConfig(const KernelConfig &cfg,
                             SCAMPPrecisionType precision);

}  // namespace SCAMP
