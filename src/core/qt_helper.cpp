#ifdef _HAS_CUDA_
#include <cuda_runtime.h>
#include <cufft.h>
#endif

#include "qt_helper.h"
#ifdef _HAS_CUDA_
#include "qt_kernels.h"
#endif

namespace SCAMP {

void qt_compute_helper::init() {
  if (_arch == CUDA_GPU_WORKER) {
#ifdef _HAS_CUDA_
    cufft_data_size = size / 2 + 1;
    if (double_precision) {
      CHECK_CUFFT_ERRORS(cufftPlan1d(&fft_plan, size, CUFFT_D2Z, 1))
      CHECK_CUFFT_ERRORS(cufftPlan1d(&ifft_plan, size, CUFFT_Z2D, 1))
    } else {
      CHECK_CUFFT_ERRORS(cufftPlan1d(&fft_plan, size, CUFFT_R2C, 1))
      CHECK_CUFFT_ERRORS(cufftPlan1d(&ifft_plan, size, CUFFT_C2R, 1))
    }
    gpuErrchk(cudaMalloc(&Q_reverse_pad, sizeof(double) * size));
    gpuErrchk(cudaMalloc(&Tc, sizeof(cuDoubleComplex) * cufft_data_size));
    gpuErrchk(cudaMalloc(&Qc, sizeof(cuDoubleComplex) * cufft_data_size));
#else
    ASSERT(false,
           "Attempted to use GPU resources in a binary not built with cuda");
#endif
  }
}

void qt_compute_helper::free() {
  if (_arch == CUDA_GPU_WORKER) {
#ifdef _HAS_CUDA_
    gpuErrchk(cudaFree(Q_reverse_pad));
    gpuErrchk(cudaFree(Tc));
    gpuErrchk(cudaFree(Qc));
    CHECK_CUFFT_ERRORS(cufftDestroy(fft_plan))
    CHECK_CUFFT_ERRORS(cufftDestroy(ifft_plan))
#else
    ASSERT(false,
           "Attempted to use GPU resources in a binary not built with cuda");
#endif
  }
}

SCAMPError_t qt_compute_helper::compute_QT_CPU(double *QT, const double *T,
                                               const double *Q) {
  const int n = size - window_size + 1;
  double rolling_sum = 0;
  double qmean = 0;
  for (int i = 0; i < window_size; ++i) {
    rolling_sum += T[i];
    qmean += Q[i];
  }
  qmean /= window_size;
  for (int i = 0; i < n; ++i) {
    double mu = rolling_sum / window_size;
    double sum = 0;
    for (int j = 0; j < window_size; ++j) {
      sum += (T[i + j] - mu) * (Q[j] - qmean);
    }
    QT[i] = sum;
    if (i != n - 1) {
      rolling_sum = rolling_sum - T[i] + T[i + window_size];
    }
  }
  return SCAMP_NO_ERROR;
}

#ifdef _HAS_CUDA_
SCAMPError_t qt_compute_helper::compute_QT(double *QT, const double *T,
                                           const double *Q,
                                           const double *qmeans,
                                           cudaStream_t s) {
  cudaError_t error;

  const int n = size - window_size + 1;

  CHECK_CUFFT_ERRORS(cufftSetStream(fft_plan, s))

  CHECK_CUFFT_ERRORS(cufftSetStream(ifft_plan, s))
  // Compute the FFT of the time series
  // For some reason the input parameter to cufftExecD2Z is not held const
  // by cufft I see nowhere in the documentation that the input vector is
  // modified using const_cast as a hack to get around this...
  CHECK_CUFFT_ERRORS(
      cufftExecD2Z(fft_plan, const_cast<double *>(T), Tc))  // NOLINT

  // clear last error
  cudaGetLastError();

  // Reverse and zero pad the query
  launch_populate_reverse_pad(Q, Q_reverse_pad, qmeans, window_size, size,
                              fft_work_size, s);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error launching populate reverse pad: %s\n",
           cudaGetErrorString(error));
    return SCAMP_CUDA_ERROR;
  }

  CHECK_CUFFT_ERRORS(cufftExecD2Z(fft_plan, Q_reverse_pad, Qc))

  launch_elementwise_multiply_inplace(Tc, Qc, cufft_data_size, fft_work_size,
                                      s);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error launching elementwise multiply inplace: %s\n",
           cudaGetErrorString(error));
    return SCAMP_CUDA_ERROR;
  }

  CHECK_CUFFT_ERRORS(cufftExecZ2D(ifft_plan, Qc, Q_reverse_pad))

  launch_normalized_aligned_dot_products(Q_reverse_pad, size, window_size, n,
                                         QT, fft_work_size, s);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error launching normalized aligned dot products: %s\n",
           cudaGetErrorString(error));
    return SCAMP_CUDA_ERROR;
  }

  return SCAMP_NO_ERROR;
}
#endif
}  // namespace SCAMP
