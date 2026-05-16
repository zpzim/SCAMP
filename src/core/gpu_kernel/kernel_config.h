#pragma once
#include <cstddef>
#include <string>
#include "common/common.h"

namespace SCAMP {

// KernelConfig holds the runtime-tunable launch parameters for the SCAMP
// do_tile<...> CUDA kernel.
//
// blocksz       -- threads per block. Tied to precision today (BLOCKSZ_SP for
//                  single/mixed, BLOCKSZ_DP for double/ultra), but exposed in
//                  the cache so future variants can vary it.
// tile_height   -- height of each tile slice processed per kernel iteration.
//                  Must match a value that the LaunchDoTile dispatch has been
//                  template-instantiated with (see kKernelVariants).
// blocks_per_sm -- hint passed to __launch_bounds__; affects register pressure
//                  vs occupancy. Like tile_height, must match an instantiated
//                  variant.
//
// diags_per_thread is not exposed here yet but the Eigen port made it
// parameterizable (info.cov is now Eigen::Array<T, DIAGS_PER_THREAD, 1>
// and the smem layout is a do_tile template param). Adding it as a 3rd
// variant axis is a planned follow-up; doing it here just means another
// case in the LaunchDoTile switch + a column in KernelVariantGeometry.
struct KernelConfig {
  int blocksz;
  int tile_height;
  int blocks_per_sm;

  bool operator==(const KernelConfig &o) const {
    return blocksz == o.blocksz && tile_height == o.tile_height &&
           blocks_per_sm == o.blocks_per_sm;
  }
};

// The defaults match the constants previously hard-coded in kernel_gpu_utils.h.
// They are the values SCAMP ships with and serve as the fallback when the
// autotune cache has no entry for the current device.
KernelConfig GetDefaultKernelConfig(SCAMPPrecisionType precision);

// Enumerated launch-geometry variants. Each variant's (tile_height,
// blocks_per_sm) tuple must have a matching branch in the LaunchDoTile
// switch in kernels_impl.h, otherwise IsSupportedKernelConfig would accept
// it but the launch would fall back to the default.
//
// The blocksz field is filled in per-precision at lookup time (via
// GetKernelConfigForVariant); only the geometry fields are enumerated here.
struct KernelVariantGeometry {
  int tile_height;
  int blocks_per_sm;
};

// Total number of enumerated variants. Defined in kernel_config.cpp.
extern const std::size_t kNumKernelVariants;

// The variant geometries themselves. Index 0 is the canonical default.
const KernelVariantGeometry &GetKernelVariantGeometry(std::size_t i);

// Materialize a full KernelConfig (including precision-derived blocksz) for
// the variant at index `i`. Asserts that i < kNumKernelVariants.
KernelConfig GetKernelConfigForVariant(std::size_t i,
                                       SCAMPPrecisionType precision);

// True iff (blocksz, tile_height, blocks_per_sm) matches any enumerated
// variant for `precision`. Used by GetKernelConfigForDevice to reject cache
// entries that name a config no current binary supports (e.g. an older or
// newer build that had a different variant set).
bool IsSupportedKernelConfig(const KernelConfig &cfg,
                             SCAMPPrecisionType precision);

}  // namespace SCAMP
