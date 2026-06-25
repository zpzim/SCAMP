#pragma once
#include <cstdint>
#include <string>

namespace SCAMP {

// Snapshot of a CUDA/HIP device's identity, used by the autotune cache so that
// configurations are scoped per-GPU-model. We deliberately key on (name +
// compute capability) rather than the device index, because the index is not
// stable across runs but the name+cap pair identifies the hardware.
struct GpuDeviceProps {
  std::string name;          // e.g. "NVIDIA GeForce RTX 3080" or "AMD Instinct MI250X"
  int compute_major;         // e.g. 8 (CUDA) or 9 (HIP gfx90a)
  int compute_minor;         // e.g. 6 (CUDA) or 0 (HIP gfx90a)
#if defined(USE_HIP)
  std::string gcn_arch_name; // e.g. "gfx90a" (HIP only)
#endif
  int sm_count;              // multiprocessor count (CU count on AMD)
  int64_t total_global_mem;  // bytes
  int max_threads_per_block;
  int max_threads_per_sm;

  // Stable string key used to look up entries in the autotune cache file.
  // Format: "<sanitized name>__sm_<major><minor>" (CUDA) or
  // "<sanitized name>__<gcn_arch>" (HIP). Sanitization is needed because
  // the raw name contains spaces.
  std::string CacheKey() const;
};

// Query the named device. Throws SCAMPException on CUDA error. Available
// only when SCAMP is built with CUDA support.
GpuDeviceProps QueryDeviceProps(int device_id);

}  // namespace SCAMP
