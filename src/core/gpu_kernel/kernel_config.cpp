#include "kernel_config.h"

#include <array>
#include <cassert>

#include "kernel_constants.h"

namespace SCAMP {

namespace {

// Enumerated launch-geometry variants. Index 0 is the canonical default.
// Each entry must have a corresponding branch in the LaunchDoTile switch
// in kernels_impl.h. Keep this short: every entry multiplies the do_tile
// template instantiation count.
constexpr std::array<KernelVariantGeometry, 2> kVariants{{
    {KERNEL_TILE_HEIGHT, BLOCKSPERSM},          // default: 256, 2
    {KERNEL_TILE_HEIGHT_ALT, BLOCKSPERSM_ALT},  // alt:     128, 4
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
  cfg.tile_height = kVariants[i].tile_height;
  cfg.blocks_per_sm = kVariants[i].blocks_per_sm;
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
    // Today blocksz isn't a variant axis; cache entries that override it
    // would land in an instantiation we don't have. (Future variants that
    // vary blocksz would extend this check.)
    return false;
  }
  for (const auto &v : kVariants) {
    if (cfg.tile_height == v.tile_height &&
        cfg.blocks_per_sm == v.blocks_per_sm) {
      return true;
    }
  }
  return false;
}

}  // namespace SCAMP
