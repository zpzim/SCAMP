#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <cinttypes>
#include <unordered_map>
#include "SCAMP.pb.h"
namespace SCAMP {

typedef union {
  float floats[2];       // floats[0] = lowest
  unsigned int ints[2];  // ints[1] = lowIdx
  uint64_t ulong;        // for atomic update
} mp_entry;

template <unsigned int count>
struct reg_mem {
  float dist[count];
  double qt[count];
};

struct OptionalArgs {
  OptionalArgs(double threshold_) : threshold(threshold_) {}

  double threshold;
};

using DeviceProfile = std::unordered_map<int, void *>;

size_t GetProfileTypeSize(SCAMPProfileType t);

enum SCAMPError_t {
  SCAMP_NO_ERROR,
  SCAMP_FUNCTIONALITY_UNIMPLEMENTED,
  SCAMP_TILE_ILLEGAL_TYPE,
  SCAMP_CUDA_ERROR,
  SCAMP_CUFFT_ERROR,
  SCAMP_CUFFT_EXEC_ERROR,
  SCAMP_DIM_INCOMPATIBLE
};

enum SCAMPTileType {
  SELF_JOIN_FULL_TILE,
  SELF_JOIN_UPPER_TRIANGULAR,
  AB_JOIN_FULL_TILE,
  AB_FULL_JOIN_FULL_TILE
};

}  // namespace SCAMP

void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
