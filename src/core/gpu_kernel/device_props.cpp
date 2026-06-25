#include "device_props.h"

#include "common/cuda_to_hip.h"
#include <sstream>
#include "common/common.h"
#include "common/scamp_exception.h"

namespace SCAMP {

namespace {

std::string SanitizeForCacheKey(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == ' ' || c == '/' || c == '|' || c == '\t' || c == '\n' ||
        c == '\r') {
      out.push_back('_');
    } else {
      out.push_back(c);
    }
  }
  return out;
}

}  // namespace

std::string GpuDeviceProps::CacheKey() const {
  std::ostringstream os;
#if defined(USE_HIP)
  // For HIP, use gcnArchName instead of compute capability
  os << SanitizeForCacheKey(name) << "__" << SanitizeForCacheKey(gcn_arch_name);
#else
  os << SanitizeForCacheKey(name) << "__sm_" << compute_major << compute_minor;
#endif
  return os.str();
}

GpuDeviceProps QueryDeviceProps(int device_id) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    throw SCAMPException(std::string("cudaGetDeviceProperties failed: ") +
                         cudaGetErrorString(err));
  }
  GpuDeviceProps out{};
  out.name = prop.name;
#if defined(USE_HIP)
  out.compute_major = prop.major;
  out.compute_minor = prop.minor;
  out.gcn_arch_name = prop.gcnArchName;
#else
  out.compute_major = prop.major;
  out.compute_minor = prop.minor;
#endif
  out.sm_count = prop.multiProcessorCount;
  out.total_global_mem = prop.totalGlobalMem;
  out.max_threads_per_block = prop.maxThreadsPerBlock;
  out.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  return out;
}

}  // namespace SCAMP
