#pragma once
#include "core/tile.h"

namespace SCAMP {

SCAMPError_t cpu_kernel_self_join_upper(Tile *t);
SCAMPError_t cpu_kernel_self_join_lower(Tile *t);
SCAMPError_t cpu_kernel_ab_join_upper(Tile *t);
SCAMPError_t cpu_kernel_ab_join_lower(Tile *t);
}  // namespace SCAMP
