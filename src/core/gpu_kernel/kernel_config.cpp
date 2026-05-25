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

namespace {

// Find the first variant of the requested family (shfl == ur==0, or
// sliding-window == ur!=0). Returns kVariants.size() if no match. By
// convention the FIRST entry of each family in SCAMP_VARIANT_TUPLES is
// the family's intended default -- variant authors should keep the
// best generalist for that family at the front of the list.
std::size_t FindFirstVariantOfFamily(bool want_shfl) {
  for (std::size_t i = 0; i < kVariants.size(); ++i) {
    if ((kVariants[i].unrolled_rows == 0) == want_shfl) return i;
  }
  return kVariants.size();
}

// Per-profile-type variant family preference, based on which family
// wins each profile in the cross-(profile,precision) autotune sweep on
// the RTX 3080 sm_86:
//   - shfl wins 1NN_INDEX + SUM_THRESH (warp-reduces atomics before
//     the final atomicMax/atomicAdd, which dominates these profiles).
//   - sliding-window wins 1NN / MATRIX_SUMMARY / AAN (light per-row
//     atomics; the SW variant's smem column buffer + heavy inner-loop
//     unroll dominates).
// Same for both precisions -- DP/SP just picks a different blocksz via
// DefaultBlockszForPrecision applied to the chosen variant.
bool ProfileTypePrefersShfl(SCAMPProfileType profile) {
  switch (profile) {
    case PROFILE_TYPE_1NN_INDEX:
    case PROFILE_TYPE_SUM_THRESH:
      return true;
    default:
      return false;
  }
}

}  // namespace

KernelConfig GetDefaultKernelConfig(SCAMPProfileType profile_type,
                                    SCAMPPrecisionType precision) {
  const bool want_shfl = ProfileTypePrefersShfl(profile_type);
  std::size_t idx = FindFirstVariantOfFamily(want_shfl);
  if (idx == kVariants.size()) {
    // Preferred family not present in this build's variant table.
    // Fall back to the other family.
    idx = FindFirstVariantOfFamily(!want_shfl);
  }
  // If kVariants is empty there's nothing we can do; v0 is at least
  // a well-defined index when at least one variant exists.
  if (idx == kVariants.size()) idx = 0;
  return GetKernelConfigForVariant(idx, precision);
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
