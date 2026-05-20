#include "kernel_config.h"

#include <array>
#include <cassert>

#include "kernel_constants.h"

namespace SCAMP {

namespace {

// Enumerated launch-geometry variants. Index 0 is the canonical default
// (matches DEFAULT_* constants from kernel_constants.h). Each entry must
// have a matching VARIANT_BRANCH in the LaunchDoTile switch in
// kernels_impl.h.
//
// Constraint: OuterUnrolledRows must be divisible by UnrolledRows (the
// outer for_<OUR/UR> loop). Derived sizes that drive register pressure:
//   inner_unrolled_cols = DPT + UR - 1   (column window held in regs)
//   unrolled_cols       = DPT + OUR - 1  (distc/idxc array width)
//
// Sentinel: unrolled_rows == 0 marks the design-A "shfl" variant. The
// per-profile dispatcher routes ur==0 entries to LaunchDoTileShflWith-
// Geometry instead of LaunchDoTileWithGeometry. ur is otherwise the inner
// row-batch size of the sliding-window kernel.
//
// Keep this short-ish -- every entry multiplies the do_tile template
// instantiation count (5 profiles x 3 precisions x 3 row/col modes x
// |kVariants|). Current: 7 variants.
constexpr std::array<KernelVariantGeometry, 7> kVariants{{
    // bps, DPT, ur, our, kti       (derived tile_height, inner_cols,
    //                               unrolled_cols)
    {DEFAULT_BLOCKSPERSM, DEFAULT_DIAGS_PER_THREAD, DEFAULT_UNROLLED_ROWS,
     DEFAULT_OUTER_UNROLLED_ROWS, DEFAULT_KERNEL_TILE_ITERS},
    // v0: 2,2,2,16,16 -> tile=256, eigen-port default sliding-window shape
    {2, 4, 2, 4, 50},
    // v1: 2,4,2,4,50  -> tile=200, DPT=4 with master-like tile height
    {2, 4, 4, 4, 50},
    // v2: 2,4,4,4,50  -> tile=200, matches pre-Eigen-port master's 4x4
    //                   hand-unroll exactly (OUR/UR=1 inner iteration)
    {4, 2, 2, 8, 16},
    // v3: 4,2,2,8,16  -> tile=128, higher occupancy + smaller tile
    {2, 2, 2, 8, 32},
    // v4: 2,2,2,8,32  -> tile=256, smaller outer-unroll = less register
    //                   pressure at the same tile height
    {1, 4, 4, 16, 16},
    // v5: 1,4,4,16,16 -> tile=256, low occupancy + big per-thread work
    {8, 4, 0, 8, 8},
    // v6: 8,4,0,8,8   -> design-A "shfl" (ur==0 sentinel), tile=64=32*DPT.
    //                   One column-block rotation per tile (the simple
    //                   case). No smem column buffer.
}};

int BlocksizeForPrecision(SCAMPPrecisionType precision) {
  switch (precision) {
    case PRECISION_ULTRA:
    case PRECISION_DOUBLE:
      return BLOCKSZ_DP;
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
  if (i == 6) {
    cfg.blocksz = 128;
  } else {
    cfg.blocksz = BlocksizeForPrecision(precision);
  }
  cfg.blocks_per_sm = kVariants[i].blocks_per_sm;
  cfg.diags_per_thread = kVariants[i].diags_per_thread;
  cfg.unrolled_rows = kVariants[i].unrolled_rows;
  cfg.outer_unrolled_rows = kVariants[i].outer_unrolled_rows;
  cfg.kernel_tile_iters = kVariants[i].kernel_tile_iters;
  return cfg;
}

KernelConfig GetDefaultKernelConfig(SCAMPPrecisionType precision) {
  // Variant 6 is the draft shfl variant being tested.
  return GetKernelConfigForVariant(6, precision);
}

bool IsSupportedKernelConfig(const KernelConfig &cfg,
                             SCAMPPrecisionType precision) {
  const auto &v = kVariants[6];
  if (cfg.blocks_per_sm == v.blocks_per_sm &&
      cfg.diags_per_thread == v.diags_per_thread &&
      cfg.unrolled_rows == v.unrolled_rows &&
      cfg.outer_unrolled_rows == v.outer_unrolled_rows &&
      cfg.kernel_tile_iters == v.kernel_tile_iters) {
    if (cfg.blocksz == 64 || cfg.blocksz == 128 || cfg.blocksz == 256 ||
        cfg.blocksz == 512) {
      return true;
    }
  }
  int expected_blocksz = BlocksizeForPrecision(precision);
  if (cfg.blocksz != expected_blocksz) {
    return false;
  }
  return false;
}

}  // namespace SCAMP
