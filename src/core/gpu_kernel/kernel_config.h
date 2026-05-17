#pragma once
#include <cstddef>
#include <string>
#include "common/common.h"

namespace SCAMP {

// KernelConfig holds the runtime-tunable launch parameters for the SCAMP
// do_tile<...> CUDA kernel. Each field except blocksz is a "variant axis"
// the autotuner can vary; blocksz is precision-tied today.
//
// blocksz             threads per block (BLOCKSZ_SP for single/mixed,
//                     BLOCKSZ_DP for double/ultra). Exposed for cache
//                     completeness; future variants could vary it.
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

// The defaults are the DEFAULT_* constants from kernel_constants.h, with
// blocksz derived from precision. Variant 0 in the table matches these.
KernelConfig GetDefaultKernelConfig(SCAMPPrecisionType precision);

// Enumerated launch-geometry variants. Each variant's full tuple must have
// a matching branch in the LaunchDoTile switch in kernels_impl.h.
//
// blocksz is filled in per-precision at lookup time (via
// GetKernelConfigForVariant); only the variable axes are enumerated here.
struct KernelVariantGeometry {
  int blocks_per_sm;
  int diags_per_thread;
  int unrolled_rows;
  int outer_unrolled_rows;
  int kernel_tile_iters;
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
