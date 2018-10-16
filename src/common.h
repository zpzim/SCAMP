#pragma once
#include <stdio.h>
#include <cinttypes>
#include <cuda_runtime.h>
namespace SCAMP {


typedef union  {
  float floats[2];                 // floats[0] = lowest
  unsigned int ints[2];                     // ints[1] = lowIdx
  uint64_t ulong;    // for atomic update
} mp_entry;

template<unsigned int count>
struct reg_mem {
    float dist[count];
    double qt[count];
};

enum FPtype {
    FP_INVALID = 0,
    FP_DOUBLE = 1,
    FP_MIXED = 2,
    FP_SINGLE = 3,
};


enum SCAMPError_t { SCAMP_NO_ERROR, SCAMP_FUNCTIONALITY_UNIMPLEMENTED, SCAMP_TILE_ILLEGAL_TYPE, SCAMP_CUDA_ERROR, SCAMP_CUFFT_ERROR, SCAMP_CUFFT_EXEC_ERROR, SCAMP_DIM_INCOMPATIBLE };

enum SCAMPTileType { SELF_JOIN_FULL_TILE, SELF_JOIN_UPPER_TRIANGULAR, AB_JOIN_FULL_TILE, AB_FULL_JOIN_FULL_TILE };


} // namespace SCAMP

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }




