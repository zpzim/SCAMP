#pragma once
#include "SCAMP.pb.h"
#include "common.h"

namespace SCAMP {

void convert_non_finite_to_zero(
    const google::protobuf::RepeatedField<double> &T, const int m,
    std::vector<double> *timeseries_clean, std::vector<bool> *nanvals);

void compute_statistics_cpu(const std::vector<double> &T,
                            const std::vector<bool> &nanvals,
                            PrecomputedInfo *info, size_t m);
}  // namespace SCAMP
