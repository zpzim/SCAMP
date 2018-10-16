#pragma once
#include <stdint.h>
#include <vector>
#include "common.h"

namespace SCAMP {

void elementwise_max_with_index(std::vector<float> &mp_full,
                                std::vector<uint32_t> &mpi_full,
                                int64_t merge_start, int64_t tile_sz,
                                std::vector<uint64_t> *to_merge);

void compute_statistics(const double *T, double *norms, double *df, double *dg,
                        double *means, size_t n, size_t m, cudaStream_t s,
                        double *scratch);

void launch_merge_mp_idx(float *mp, uint32_t *mpi, uint32_t n, uint64_t *merged,
                         cudaStream_t s);

}  // namespace SCAMP
