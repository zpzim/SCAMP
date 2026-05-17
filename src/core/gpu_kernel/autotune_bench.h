// Header for the default BenchmarkFn implementation in autotune_bench.cpp.
// Callers that want the default synthetic-self-join workload pass
// DefaultBenchmarkVariant as the bench arg to RunAutotuneWithBenchmark.
//
// Lives in a separate header (and TU) from autotune.h because the
// implementation pulls in scamp_op via <common/scamp_interface.h>;
// gpu_utils (which owns autotune.h) deliberately doesn't depend on
// scamp_op to avoid a cyclic static-library link.
#pragma once

#include "autotune.h"
#include "common/common.h"
#include "kernel_config.h"

namespace SCAMP {

// Runs a small synthetic self-join (65K x 200-window random Normal
// vector) on `device_id` using the requested (profile, precision) and
// the variant `cfg`, returning wall-clock seconds. Throws on failure.
// Suitable as the BenchmarkFn argument to RunAutotuneWithBenchmark.
double DefaultBenchmarkVariant(int device_id, SCAMPProfileType profile,
                               SCAMPPrecisionType precision,
                               const KernelConfig &cfg);

}  // namespace SCAMP
