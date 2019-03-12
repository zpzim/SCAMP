#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_helper.h"
#include "fft_kernels.h"

namespace SCAMP {

SCAMPError_t fft_precompute_helper::compute_QT(double *QT, const double *T,
                                               const double *Q,
                                               const double *qmeans,
                                               cudaStream_t s) {
  cufftResult cufftError;
  cudaError_t error;

  const int n = size - window_size + 1;

  cufftError = cufftSetStream(fft_plan, s);

  if (cufftError != CUFFT_SUCCESS) {
    CHECK_CUFFT_ERRORS(cufftError);
    return SCAMP_CUFFT_ERROR;
  }
  cufftError = cufftSetStream(ifft_plan, s);
  if (cufftError != CUFFT_SUCCESS) {
    CHECK_CUFFT_ERRORS(cufftError);
    return SCAMP_CUFFT_ERROR;
  }
  // Compute the FFT of the time series
  // For some reason the input parameter to cufftExecD2Z is not held const by
  // cufft
  // I see nowhere in the documentation that the input vector is modified
  // using const_cast as a hack to get around this...
  cufftError = cufftExecD2Z(fft_plan, const_cast<double *>(T), Tc);  // NOLINT

  printf("T = %p\n", T); 
  printf("Tc = %p\n", Tc);
  printf("n = %d\n", n);
  if (cufftError != CUFFT_SUCCESS) {
    CHECK_CUFFT_ERRORS(cufftError);
    return SCAMP_CUFFT_EXEC_ERROR;
  }

  // Reverse and zero pad the query
  launch_populate_reverse_pad(Q, Q_reverse_pad, qmeans, window_size, size,
                              fft_work_size, s);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return SCAMP_CUDA_ERROR;
  }

  cufftError = cufftExecD2Z(fft_plan, Q_reverse_pad, Qc);
  if (cufftError != CUFFT_SUCCESS) {
    CHECK_CUFFT_ERRORS(cufftError);
    return SCAMP_CUFFT_EXEC_ERROR;
  }

  launch_elementwise_multiply_inplace(Tc, Qc, cufft_data_size, fft_work_size,
                                      s);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return SCAMP_CUDA_ERROR;
  }

  cufftError = cufftExecZ2D(ifft_plan, Qc, Q_reverse_pad);

  if (cufftError != CUFFT_SUCCESS) {
    CHECK_CUFFT_ERRORS(cufftError);
    return SCAMP_CUFFT_EXEC_ERROR;
  }
  launch_normalized_aligned_dot_products(Q_reverse_pad, size, window_size, n,
                                         QT, fft_work_size, s);
  error = cudaPeekAtLastError();

  if (error != cudaSuccess) {
    return SCAMP_CUDA_ERROR;
  }

  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
