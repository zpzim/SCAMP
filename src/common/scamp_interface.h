#pragma once

#include "scamp_args.h"

#include <vector>

namespace SCAMP {

// SCAMP library interface
void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices,
              int num_threads);

void do_SCAMP(SCAMPArgs *args);

int num_available_gpus();

}  // namespace SCAMP
