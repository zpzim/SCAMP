#include "kernel_config.h"

#include <array>
#include <cassert>

#include "kernel_constants.h"

namespace SCAMP {

namespace {

// Enumerated launch-geometry variants. Index 0 is the canonical default
// (matches DEFAULT_* constants from kernel_constants.h). Each entry must
// have a matching branch in the LaunchDoTile switch in kernels_impl.h.
//
// Keep this short -- every entry multiplies the do_tile template
// instantiation count (5 profiles x 3 precisions x 3 row/col modes x
// |kVariants|).
constexpr std::array<KernelVariantGeometry, 2> kVariants{{
    // bps, DPT, ur, our, kti       (derived tile_height)
    {DEFAULT_BLOCKSPERSM, DEFAULT_DIAGS_PER_THREAD, DEFAULT_UNROLLED_ROWS,
     DEFAULT_OUTER_UNROLLED_ROWS, DEFAULT_KERNEL_TILE_ITERS},  // 2,2,2,16,16 (256)
    {2, 4, 2, 4, 50},                                          // 2,4,2,4,50  (200)
}};

int BlocksizeForPrecision(SCAMPPrecisionType precision) {
  switch (precision) {
    case PRECISION_ULTRA:
    case PRECISION_DOUBLE:
      return BLOCKSZ_DP;
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      return BLOCKSZ_SP;
    default:
      return BLOCKSZ_DP;
  }
}

}  // namespace

const std::size_t kNumKernelVariants = kVariants.size();

const KernelVariantGeometry &GetKernelVariantGeometry(std::size_t i) {
  assert(i < kVariants.size());
  return kVariants[i];
}

KernelConfig GetKernelConfigForVariant(std::size_t i,
                                       SCAMPPrecisionType precision) {
  assert(i < kVariants.size());
  KernelConfig cfg{};
  cfg.blocksz = BlocksizeForPrecision(precision);
  cfg.blocks_per_sm = kVariants[i].blocks_per_sm;
  cfg.diags_per_thread = kVariants[i].diags_per_thread;
  cfg.unrolled_rows = kVariants[i].unrolled_rows;
  cfg.outer_unrolled_rows = kVariants[i].outer_unrolled_rows;
  cfg.kernel_tile_iters = kVariants[i].kernel_tile_iters;
  return cfg;
}

KernelConfig GetDefaultKernelConfig(SCAMPPrecisionType precision) {
  // Variant 0 is the canonical default by convention.
  return GetKernelConfigForVariant(0, precision);
}

bool IsSupportedKernelConfig(const KernelConfig &cfg,
                             SCAMPPrecisionType precision) {
  int expected_blocksz = BlocksizeForPrecision(precision);
  if (cfg.blocksz != expected_blocksz) {
    return false;
  }
  for (const auto &v : kVariants) {
    if (cfg.blocks_per_sm == v.blocks_per_sm &&
        cfg.diags_per_thread == v.diags_per_thread &&
        cfg.unrolled_rows == v.unrolled_rows &&
        cfg.outer_unrolled_rows == v.outer_unrolled_rows &&
        cfg.kernel_tile_iters == v.kernel_tile_iters) {
      return true;
    }
  }
  return false;
}

}  // namespace SCAMP
