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

struct PrecomputedInfo {
private:
  std::vector<double> _norms;
  std::vector<double> _df;
  std::vector<double> _dg;
  std::vector<double> _means;
public:
  const std::vector<double>& dg() const { return _dg; }
  const std::vector<double>& df() const { return _df; }
  const std::vector<double>& norms() const { return _norms; }
  const std::vector<double>& means() const { return _means; }
  std::vector<double>& mutable_dg() { return _dg; }
  std::vector<double>& mutable_df() { return _df; }
  std::vector<double>& mutable_norms()  { return _norms; }
  std::vector<double>& mutable_means()  { return _means; }
  
};

using DeviceProfile = std::unordered_map<int, void *>;

size_t GetProfileTypeSize(SCAMPProfileType t);

enum SCAMPArchitecture {
  CPU_WORKER,
  CUDA_GPU_WORKER,
};

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
