#include "kernel_config.h"

#include <array>
#include <cassert>

// Generated from SCAMP_VARIANT_TUPLES in CMakeLists.txt. Provides the
// kVariants[] constexpr array at namespace SCAMP scope.
#include "kernel_variants_table.h"

namespace SCAMP {

// Enumerated launch-geometry variants. Each entry must have a matching
// VARIANT_BRANCH in the LaunchDoTile switch in kernels_impl.h.
//
// Constraint: OuterUnrolledRows must be divisible by UnrolledRows (the
// outer for_<OUR/UR> loop). Derived sizes that drive register pressure:
//   inner_unrolled_cols = DPT + UR - 1   (column window held in regs)
//   unrolled_cols       = DPT + OUR - 1  (distc/idxc array width)
//
// Sentinel: unrolled_rows == 0 marks the "shfl" (cov-shuffle) variant.
// The per-profile dispatcher routes ur==0 entries to
// LaunchDoTileShflWithGeometry instead of LaunchDoTileWithGeometry.
// ur is otherwise the inner row-batch size of the sliding-window kernel.
//
// The kVariants[] array is generated from SCAMP_VARIANT_TUPLES in
// src/core/gpu_kernel/CMakeLists.txt -- single source of truth for
// which variant geometries the binary supports. See
// kernel_variants_table.h.in for the @-substitution template.

const std::size_t kNumKernelVariants = kVariants.size();

const KernelVariantGeometry &GetKernelVariantGeometry(std::size_t i) {
  assert(i < kVariants.size());
  return kVariants[i];
}

int DefaultBlockszForPrecision(const KernelVariantGeometry &v,
                               SCAMPPrecisionType precision) {
  // DP / ULTRA target threads/SM is half of SP (DP = 2x registers per
  // thread). The variant tuple stores only default_blocksz_sp; the DP
  // default is derived here so the 2:1 ratio is enforced by construction
  // (impossible to violate via a malformed tuple).
  switch (precision) {
    case PRECISION_ULTRA:
    case PRECISION_DOUBLE:
      return v.default_blocksz_sp / 2;
    case PRECISION_SINGLE:
      return v.default_blocksz_sp;
    default:
      return v.default_blocksz_sp / 2;
  }
}

KernelConfig GetKernelConfigForVariant(std::size_t i,
                                       SCAMPPrecisionType precision) {
  assert(i < kVariants.size());
  KernelConfig cfg{};
  // Cold-start blocksz comes from the variant's SP default, halved for DP.
  // The autotune sweep also tries the other three values in
  // {64,128,256,512} via the independent blocksz axis, and the on-disk
  // cache can store any of them; these defaults only feed the path
  // where no cache entry exists for this device + (profile, precision).
  cfg.blocksz = DefaultBlockszForPrecision(kVariants[i], precision);
  cfg.blocks_per_sm = kVariants[i].blocks_per_sm;
  cfg.diags_per_thread = kVariants[i].diags_per_thread;
  cfg.unrolled_rows = kVariants[i].unrolled_rows;
  cfg.outer_unrolled_rows = kVariants[i].outer_unrolled_rows;
  cfg.kernel_tile_iters = kVariants[i].kernel_tile_iters;
  return cfg;
}

KernelConfig GetDefaultKernelConfig(SCAMPPrecisionType precision) {
  // The shfl variant with the smallest tile is the safest default: it
  // has the tightest worst-case ratio in the cross-target score table
  // on every GPU we've measured, and uses the least smem so it fits
  // cleanly even on older devices. By convention this is the first
  // shfl (ur==0) entry in SCAMP_VARIANT_TUPLES.
  for (std::size_t i = 0; i < kVariants.size(); ++i) {
    if (kVariants[i].unrolled_rows == 0) {
      return GetKernelConfigForVariant(i, precision);
    }
  }
  // No shfl variant configured -- fall back to v0.
  return GetKernelConfigForVariant(0, precision);
}

bool IsSupportedKernelConfig(const KernelConfig &cfg,
                             SCAMPPrecisionType precision) {
  (void)precision;
  // Both the sliding-window and shfl LaunchDoTile helpers dispatch by
  // runtime blocksz, so the autotuner is free to sweep that axis for
  // any variant tuple in kVariants.
  for (const auto &v : kVariants) {
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
