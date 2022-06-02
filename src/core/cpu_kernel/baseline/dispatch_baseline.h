#pragma once

#include "core/tile.h"
#include "core/kernel_common.h"

namespace SCAMP {

SCAMPError_t dispatch_kernel_baseline(SCAMPKernelInputArgs<double> args,
                             Tile *t, void *profile_a,
                             void *profile_b, bool do_rows,
                             bool do_cols);
}