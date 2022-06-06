#pragma once

#include "core/kernel_common.h"
#include "core/tile.h"

namespace SCAMP {

#if defined(_SCAMP_USE_AVX_)
namespace AVX {
#elif defined(_SCAMP_USE_AVX2_)
namespace AVX2 {
#else
namespace BASELINE {
#endif

SCAMPError_t compute_cpu_resources_and_launch(
    SCAMP::SCAMPKernelInputArgs<double> args, SCAMP::Tile *t, void *profile_a,
    void *profile_b, bool do_rows, bool do_cols);
}

}  // namespace SCAMP
