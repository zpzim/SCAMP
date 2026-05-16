// Top-level GPU kernel dispatcher and self/ab-join entry points.
//
// The heavy templated kernel body (do_tile<...>) and its precision/row-col
// dispatcher (LaunchDoTile<...>) live in kernels_impl.h and are instantiated
// once per profile type in kernel_<profile>.cu. This translation unit only
// includes the slim kernels_dispatch.h, so it compiles quickly and does not
// duplicate the kernel instantiations.
#include <cub/device/device_merge_sort.cuh>

#include "autotune.h"
#include "core/defines.h"
#include "core/kernel_common.h"
#include "kernel_config.h"
#include "kernel_gpu_utils.h"
#include "kernels.h"
#include "kernels_dispatch.h"

namespace SCAMP {

SCAMPError_t compute_gpu_resources_and_launch(SCAMPKernelInputArgs<double> args,
                                              Tile *t, void *profile_a,
                                              void *profile_b, bool do_rows,
                                              bool do_cols) {
  int exclusion_total = args.exclusion_lower + args.exclusion_upper;
  // Pull the per-device kernel config from the autotune cache. The returned
  // cfg names a (blocksz, tile_height, blocks_per_sm) tuple that
  // IsSupportedKernelConfig has already validated against the enumerated
  // variant table (see kernel_config.cpp). LaunchKernel_<X> below switches
  // on the cfg to pick the matching pre-instantiated kernel variant.
  KernelConfig cfg = GetKernelConfigForDevice(
      t->get_cuda_id(), t->info()->profile_type, t->info()->fp_type);
  uint64_t blocksz = cfg.blocksz;
  uint64_t num_workers = ceil((args.n_x - exclusion_total) /
                              static_cast<double>(DIAGS_PER_THREAD));
  uint64_t num_blocks = ceil(num_workers / static_cast<double>(blocksz));
  uint64_t smem = get_smem(t->info(), blocksz, cfg.tile_height);
  if (!t->info()->silent_mode) {
    std::cout << "Launching " << num_blocks << " thread blocks of size "
              << blocksz << " (tile_height=" << cfg.tile_height
              << ", blocks_per_sm=" << cfg.blocks_per_sm << ") with a total of "
              << smem << " bytes of shared memory per block." << std::endl;
  }
  if (exclusion_total >= args.n_x) {
    return SCAMP_NO_ERROR;
  }
  switch (t->info()->profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
      return LaunchKernel_SUM_THRESH(
          args, reinterpret_cast<double *>(profile_a),
          reinterpret_cast<double *>(profile_b), t->info()->fp_type, do_rows,
          do_cols, cfg, num_blocks, smem, t->get_stream());
    case PROFILE_TYPE_1NN_INDEX:
      return LaunchKernel_1NN_INDEX(
          args, reinterpret_cast<uint64_t *>(profile_a),
          reinterpret_cast<uint64_t *>(profile_b), t->info()->fp_type, do_rows,
          do_cols, cfg, num_blocks, smem, t->get_stream());
    case PROFILE_TYPE_1NN:
      return LaunchKernel_1NN(args, reinterpret_cast<float *>(profile_a),
                              reinterpret_cast<float *>(profile_b),
                              t->info()->fp_type, do_rows, do_cols, cfg,
                              num_blocks, smem, t->get_stream());
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      return LaunchKernel_APPROX_ALL_NEIGHBORS(
          args, reinterpret_cast<SCAMPmatch *>(profile_a),
          reinterpret_cast<SCAMPmatch *>(profile_b), t->info()->fp_type,
          do_rows, do_cols, cfg, num_blocks, smem, t->get_stream());
    case PROFILE_TYPE_MATRIX_SUMMARY:
      return LaunchKernel_MATRIX_SUMMARY(
          args, reinterpret_cast<float *>(profile_a),
          reinterpret_cast<float *>(profile_b), t->info()->fp_type, do_rows,
          do_cols, cfg, num_blocks, smem, t->get_stream());
    default:
      return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }
}

SCAMPError_t gpu_kernel_self_join_upper(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, false, false);
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t gpu_kernel_self_join_lower(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, true, false);
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}

SCAMPError_t gpu_kernel_ab_join_upper(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, false, true);
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t gpu_kernel_ab_join_lower(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, true, true);
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}

// Functor wrapping SCAMPmatch::operator< so it's usable from CUB device code.
// HOST_DEVICE_FUNCTION expands to __host__ __device__.
struct SCAMPmatchLess {
  HOST_DEVICE_FUNCTION bool operator()(const SCAMPmatch &a,
                                       const SCAMPmatch &b) const {
    return a < b;
  }
};

void match_gpu_sort(SCAMPmatch *matches, int64_t len, cudaStream_t stream) {
  // CUB DeviceMergeSort follows the standard two-call pattern: a first call
  // with d_temp_storage = nullptr reports the required scratch size, then we
  // allocate that scratch and call again to actually sort.  We use
  // cudaMallocAsync / cudaFreeAsync (CUDA 11.2+) so the whole pipeline stays
  // on the user-supplied stream.
  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceMergeSort::SortKeys(d_temp, temp_bytes, matches, len,
                                 SCAMPmatchLess(), stream);
  if (temp_bytes > 0) {
    cudaMallocAsync(&d_temp, temp_bytes, stream);
  }
  cub::DeviceMergeSort::SortKeys(d_temp, temp_bytes, matches, len,
                                 SCAMPmatchLess(), stream);
  if (d_temp != nullptr) {
    cudaFreeAsync(d_temp, stream);
  }
}

}  // namespace SCAMP
