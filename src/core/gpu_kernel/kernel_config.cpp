#include "kernel_config.h"
#include "kernel_constants.h"

namespace SCAMP {

KernelConfig GetDefaultKernelConfig(SCAMPPrecisionType precision) {
  KernelConfig cfg{};
  cfg.tile_height = KERNEL_TILE_HEIGHT;
  cfg.blocks_per_sm = BLOCKSPERSM;
  switch (precision) {
    case PRECISION_ULTRA:
    case PRECISION_DOUBLE:
      cfg.blocksz = BLOCKSZ_DP;
      break;
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      cfg.blocksz = BLOCKSZ_SP;
      break;
    default:
      cfg.blocksz = BLOCKSZ_DP;
      break;
  }
  return cfg;
}

bool IsSupportedKernelConfig(const KernelConfig &cfg,
                             SCAMPPrecisionType precision) {
  return cfg == GetDefaultKernelConfig(precision);
}

}  // namespace SCAMP
