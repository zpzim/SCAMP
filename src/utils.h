#pragma once
#include <stdint.h>
#include <vector>
#include "SCAMP.pb.h"
#include "common.h"

namespace SCAMP {

void compute_statistics(const google::protobuf::RepeatedField<double> &T, PrecomputedInfo *info, size_t m);

void launch_merge_mp_idx(float *mp, uint32_t *mpi, uint32_t n, uint64_t *merged,
                         cudaStream_t s);

}  // namespace SCAMP
