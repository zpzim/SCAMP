#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_helper.h"
#include "fft_kernels.h"

namespace SCAMP {

SCAMPError_t fft_precompute_helper::compute_QT(double *QT, const double *T,
                                               const double *Q,
                                               const double *qmeans,
                                               cudaStream_t s) {
  cudaError_t error;

  const int n = size - window_size + 1;

  CHECK_CUFFT_ERRORS(cufftSetStream(fft_plan, s));

  CHECK_CUFFT_ERRORS(cufftSetStream(ifft_plan, s));
  // Compute the FFT of the time series
  // For some reason the input parameter to cufftExecD2Z is not held const by
  // cufft
  // I see nowhere in the documentation that the input vector is modified
  // using const_cast as a hack to get around this...
  CHECK_CUFFT_ERRORS(
      cufftExecD2Z(fft_plan, const_cast<double *>(T), Tc));  // NOLINT

  // Reverse and zero pad the query
  launch_populate_reverse_pad(Q, Q_reverse_pad, qmeans, window_size, size,
                              fft_work_size, s);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return SCAMP_CUDA_ERROR;
  }

  CHECK_CUFFT_ERRORS(cufftExecD2Z(fft_plan, Q_reverse_pad, Qc));

  launch_elementwise_multiply_inplace(Tc, Qc, cufft_data_size, fft_work_size,
                                      s);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return SCAMP_CUDA_ERROR;
  }

  CHECK_CUFFT_ERRORS(cufftExecZ2D(ifft_plan, Qc, Q_reverse_pad));

  launch_normalized_aligned_dot_products(Q_reverse_pad, size, window_size, n,
                                         QT, fft_work_size, s);
  error = cudaPeekAtLastError();

  if (error != cudaSuccess) {
    return SCAMP_CUDA_ERROR;
  }

  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
