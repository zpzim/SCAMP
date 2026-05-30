#pragma once
#include <cstdint>
#include <string>

namespace SCAMP {

// Snapshot of a CUDA device's identity, used by the autotune cache so that
// configurations are scoped per-GPU-model. We deliberately key on (name +
// compute capability) rather than the device index, because the index is not
// stable across runs but the name+cap pair identifies the hardware.
struct GpuDeviceProps {
  std::string name;          // e.g. "NVIDIA GeForce RTX 3080"
  int compute_major;         // e.g. 8
  int compute_minor;         // e.g. 6
  int sm_count;              // multiprocessor count
  int64_t total_global_mem;  // bytes
  int max_threads_per_block;
  int max_threads_per_sm;

  // Stable string key used to look up entries in the autotune cache file.
  // Format: "<sanitized name>__sm_<major><minor>" e.g.
  // "NVIDIA_GeForce_RTX_3080__sm_86". Sanitization is needed because the
  // raw name contains spaces.
  std::string CacheKey() const;
};

// Query the named device. Throws SCAMPException on CUDA error. Available
// only when SCAMP is built with CUDA support.
GpuDeviceProps QueryDeviceProps(int device_id);

}  // namespace SCAMP
