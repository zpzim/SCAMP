#pragma once

#include "core/kernel_common.h"
#include "core/tile.h"

namespace SCAMP {

SCAMPError_t dispatch_kernel_avx2(SCAMPKernelInputArgs<double> args, Tile *t,
                                  void *profile_a, void *profile_b,
                                  bool do_rows, bool do_cols);
}
