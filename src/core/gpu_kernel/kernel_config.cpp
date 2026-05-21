#include "kernel_config.h"

#include <array>
#include <cassert>

#include "kernel_constants.h"
// Generated from SCAMP_VARIANT_TUPLES in CMakeLists.txt. Provides the
// kVariants[] constexpr array at namespace SCAMP scope.
#include "kernel_variants_table.h"

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
// The kVariants[] array is generated from SCAMP_VARIANT_TUPLES in
// src/core/gpu_kernel/CMakeLists.txt. Adding, removing, or reordering
// variants is a single-source-of-truth edit there -- the include
// below is the only place this file consumes the table. See
// kernel_variants_table.h.in for the @-substitution template.
//
// Historical note: an earlier iteration had 9 entries (v0..v8).
// v1, v3, v4, v7 (pre-prune labels) were retired after the multi-
// device autotune sweep showed they never won any (profile,
// precision) target, then the labels were compacted to 0..4.

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
  // shfl variants (ur==0 sentinel) default to blocksz=128; the autotune
  // sweep will also try 64/256/512 via the blocksz axis. Sliding-window
  // variants default to the precision-tied BLOCKSZ_DP/BLOCKSZ_SP.
  if (kVariants[i].unrolled_rows == 0) {
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
  // v3 (the shfl variant with the smallest tile -- pre-prune label v6)
  // is the safest default: it wins one 3080 target outright, has the
  // tightest worst-case ratio in the cross-target score table on every
  // GPU we've measured, and uses the least smem so it fits cleanly even
  // on older devices.
  return GetKernelConfigForVariant(3, precision);
}

bool IsSupportedKernelConfig(const KernelConfig &cfg,
                             SCAMPPrecisionType precision) {
  (void)precision;
  // All of kVariants is currently enabled (the table is the curated set
  // -- the pre-prune holes for v1/v3/v4/v7 were compacted out). Each
  // variant accepts any of the standard blocksz values -- both the
  // sliding-window and shfl LaunchDoTile helpers dispatch by runtime
  // blocksz so the autotuner can sweep that axis too.
  constexpr int kEnabledVariants[] = {0, 1, 2, 3, 4};
  for (int idx : kEnabledVariants) {
    const auto &v = kVariants[idx];
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
  }
  return false;
}

}  // namespace SCAMP
