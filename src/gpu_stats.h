#pragma once
#include "SCAMP.pb.h"
#include "common.h"

namespace SCAMP {

void compute_statistics_gpu(const google::protobuf::RepeatedField<double> &T,
                            PrecomputedInfo *info, size_t m);

}  // namespace SCAMP
