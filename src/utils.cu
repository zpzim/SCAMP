#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "utils.h"
namespace SCAMP {

// This kernel computes a sliding mean with specified window size and a
// corresponding prefix sum array (A)
__global__ void sliding_mean(double *pref_sum, size_t window, size_t size,
                             double *means) {
  const double coeff = 1.0 / (double)window;
  size_t a = blockIdx.x * blockDim.x + threadIdx.x;
  size_t b = blockIdx.x * blockDim.x + threadIdx.x + window;

  if (a == 0) {
    means[a] = pref_sum[window - 1] * coeff;
  }
  if (a < size - 1) {
    means[a + 1] = (pref_sum[b] - pref_sum[a]) * coeff;
  }
}

__global__ void sliding_norm(double *cumsumsqr, unsigned int window,
                             unsigned int size, double *norms) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.x * blockDim.x + threadIdx.x + window;
  if (a == 0) {
    norms[a] = 1 / sqrt(cumsumsqr[window - 1]);
  } else if (b < size + window) {
    norms[a] = 1 / sqrt(cumsumsqr[b - 1] - cumsumsqr[a - 1]);
  }
}

__global__ void sliding_dfdg(const double *T, const double *means, double *df,
                             double *dg, const int m, const int n) {
  const double half = 1.0 / (double)2.0;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n - 1) {
    df[tid] = (T[tid + m] - T[tid]) * half;
    dg[tid] = (T[tid + m] - means[tid + 1]) + (T[tid] - means[tid]);
  }
}

__global__ void __launch_bounds__(512, 4)
    fastinvnorm(double *norm, const double *mean, const double *T, int m,
                int n) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int jump = ceil(n / (double)(blockDim.x * gridDim.x));
  int start = jump * tid;
  int end = jump * (tid + 1);
  end = min(end, n);
  if (start >= n) {
    return;
  }
  double sum = 0;
  for (int i = 0; i < m; ++i) {
    double val = T[i + start] - mean[start];
    sum += val * val;
  }
  norm[start] = sum;

  for (int i = start + 1; i < end; ++i) {
    norm[i] = norm[i - 1] +
              ((T[i - 1] - mean[i - 1]) + (T[i + m - 1] - mean[i])) *
                  (T[i + m - 1] - T[i - 1]);
  }
  for (int i = start; i < end; ++i) {
    norm[i] = 1.0 / sqrt(norm[i]);
  }
}

__global__ void cross_correlation_to_ed(float *profile, unsigned int n,
                                        unsigned int m) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    profile[tid] = sqrt(max(2 * (1 - profile[tid]), 0.0)) * sqrt((double)m);
  }
}

__global__ void merge_mp_idx(float *mp, uint32_t *mpi, uint32_t n,
                             unsigned long long *merged) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    mp_entry item;
    item.floats[0] = (float)mp[tid];
    item.ints[1] = mpi[tid];
    merged[tid] = item.ulong;
  }
}

void elementwise_max_with_index(std::vector<float> &mp_full,
                                std::vector<uint32_t> &mpi_full,
                                int64_t merge_start, int64_t tile_sz,
                                std::vector<unsigned long long> *to_merge) {
  printf("In elementwise max\n");
  for (int i = 0; i < tile_sz; ++i) {
    mp_entry curr;
    curr.ulong = to_merge->at(i);
    if (mp_full[i + merge_start] < curr.floats[0]) {
      mp_full[i + merge_start] = curr.floats[0];
      mpi_full[i + merge_start] = curr.ints[1];
    }
  }
  printf("Done elementwise max\n");
}

void compute_statistics(const double *T, double *norms, double *df, double *dg,
                        double *means, size_t n, size_t m, cudaStream_t s,
                        double *scratch) {
  dim3 grid(ceil(n / (double)512), 1, 1);
  dim3 block(512, 1, 1);
  thrust::device_ptr<const double> dev_ptr_T =
      thrust::device_pointer_cast<const double>(T);
  thrust::device_ptr<double> dev_ptr_scratch =
      thrust::device_pointer_cast<double>(scratch);
  thrust::inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T,
                         dev_ptr_T + n + m - 1, dev_ptr_scratch,
                         thrust::plus<double>());
  // cub::DeviceScan::InclusiveSum(temp, bytes, T, scratch, n + m - 1, s);
  // Allocate temporary storage
  // cudaMalloc(&temp, bytes);
  // cub::DeviceScan::InclusiveSum(temp, bytes, T, scratch, n + m - 1, s);
  // cudaFree(temp);
  // prefix_sum(T, n+m-1, scratch, s);
  gpuErrchk(cudaPeekAtLastError());
  // Use prefix sum to compute sliding mean
  sliding_mean<<<grid, block, 0, s>>>(scratch, m, n, means);
  gpuErrchk(cudaPeekAtLastError());

  // Compute differential values
  sliding_dfdg<<<grid, block, 0, s>>>(T, means, df, dg, m, n);
  gpuErrchk(cudaPeekAtLastError());

  // This will be kind of slow on the GPU, may cause latency between tiles
  int workers = n / m + 1;
  fastinvnorm<<<dim3(ceil(workers / (double)512), 1, 1), dim3(512, 1, 1), 0,
                s>>>(norms, means, T, m, n);
  gpuErrchk(cudaPeekAtLastError());
}

void launch_merge_mp_idx(float *mp, uint32_t *mpi, uint32_t n,
                         unsigned long long *merged, cudaStream_t s) {
  merge_mp_idx<<<dim3(std::ceil(n / 1024.0), 1, 1), dim3(1024, 1, 1), 0, s>>>(
      mp, mpi, n, merged);
}

}  // namespace SCAMP
