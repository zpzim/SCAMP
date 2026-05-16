#pragma once
#include <string>
#include "common/common.h"

namespace SCAMP {

// KernelConfig holds the runtime-tunable launch parameters for the SCAMP
// do_tile<...> CUDA kernel.
//
// blocksz       -- threads per block (must match a value the kernel was
//                  template-instantiated with; today only the defaults are
//                  instantiated, future variants will go in kernels.cu)
// tile_height   -- height of each tile slice processed per kernel iteration
//                  (must match an instantiated variant for the same reason)
// blocks_per_sm -- hint passed to __launch_bounds__; affects register pressure
//                  vs occupancy tradeoff
//
// diags_per_thread is intentionally not exposed here: the kernel hard-codes
// it to 4 via cov1..cov4 register names. Changing it requires a kernel
// rewrite, not a runtime knob.
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
// They are the values SCAMP has been shipping with and serve as the fallback
// when the autotune cache has no entry for the current device.
KernelConfig GetDefaultKernelConfig(SCAMPPrecisionType precision);

// True iff (blocksz, tile_height, blocks_per_sm) is a triple the kernel has
// been template-instantiated for. Today only the defaults qualify; this hook
// lets follow-up PRs add variants without changing the cache loader.
bool IsSupportedKernelConfig(const KernelConfig &cfg,
                             SCAMPPrecisionType precision);

}  // namespace SCAMP
