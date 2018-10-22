#pragma once
#include <stdint.h>
#include <vector>
#include "SCAMP.pb.h"
#include "common.h"

namespace SCAMP {

void compute_statistics(const double *T, double *norms, double *df, double *dg,
                        double *means, size_t n, size_t m, cudaStream_t s,
                        double *scratch);

void launch_merge_mp_idx(float *mp, uint32_t *mpi, uint32_t n, uint64_t *merged,
                         cudaStream_t s);

}  // namespace SCAMP
