#include "qt_kernels.h"

__global__ void elementwise_multiply_inplace(const cuDoubleComplex *A,
                                             cuDoubleComplex *B,
                                             const int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    B[tid] = cuCmul(A[tid], B[tid]);
  }
}

// A is input unaligned sliding dot products produced by ifft
// out is the computed vector of distances
__global__ void normalized_aligned_dot_products(const double *A,
                                                const double divisor,
                                                const unsigned int m,
                                                const unsigned int n,
                                                double *QT) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a < n) {
    QT[a] = A[a + m - 1] / divisor;
  }
}

__global__ void populate_reverse_pad(const double *Q, double *Q_reverse_pad,
                                     const double *mean, const int window_size,
                                     const int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double mu = *mean;
  if (tid < window_size) {
    Q_reverse_pad[tid] = Q[window_size - 1 - tid] - mu;
  } else if (tid < size) {
    Q_reverse_pad[tid] = 0;
  }
}

void launch_populate_reverse_pad(const double *Q, double *Q_reverse_pad,
                                 const double *mean, const int window_size,
                                 const int size, int fft_work_size,
                                 cudaStream_t s) {
  dim3 block(fft_work_size, 1, 1);
  populate_reverse_pad<<<dim3(ceil(size / (float)fft_work_size), 1, 1), block,
                         0, s>>>(Q, Q_reverse_pad, mean, window_size, size);
}

void launch_elementwise_multiply_inplace(const cuDoubleComplex *A,
                                         cuDoubleComplex *B, const int size,
                                         int fft_work_size, cudaStream_t s) {
  dim3 block(fft_work_size, 1, 1);
  elementwise_multiply_inplace<<<dim3(ceil(size / (float)fft_work_size), 1, 1),
                                 block, 0, s>>>(A, B, size);
}
void launch_normalized_aligned_dot_products(const double *A,
                                            const double divisor,
                                            const unsigned int m,
                                            const unsigned int n, double *QT,
                                            int fft_work_size, cudaStream_t s) {
  dim3 block(fft_work_size, 1, 1);
  normalized_aligned_dot_products<<<dim3(ceil(n / (float)fft_work_size), 1, 1),
                                    block, 0, s>>>(A, divisor, m, n, QT);
}
