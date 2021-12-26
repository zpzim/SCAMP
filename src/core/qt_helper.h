#pragma once

#ifdef _HAS_CUDA_
#include <cufft.h>
#endif

#include <stdlib.h>
#include "common/common.h"
#include "common/scamp_exception.h"

#ifdef _CUFFT_H_

// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";
    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    default:
      return "CUFFT UNKNOWN ERROR";
  }
}

#define CHECK_CUFFT_ERRORS(call)                                        \
  {                                                                     \
    cufftResult_t err;                                                  \
    if ((err = (call)) != CUFFT_SUCCESS) {                              \
      std::ostringstream ostream;                                       \
      ostream << "cuFFT error " << err << ":" << _cudaGetErrorEnum(err) \
              << " at " << __FILE__ << ":" << __LINE__;                 \
      throw SCAMPException(ostream.str());                              \
    }                                                                   \
  }
#endif

namespace SCAMP {

class qt_compute_helper {
 private:
  const size_t size;
  const size_t window_size;
  const bool double_precision;
  const SCAMPArchitecture _arch;
// CUFFT specific variables
#ifdef _HAS_CUDA_
  double *Q_reverse_pad;
  cuDoubleComplex *Qc, *Tc;
  cufftHandle fft_plan, ifft_plan;
  size_t cufft_data_size;
  const int fft_work_size = 512;
#endif

  void init();
  void free();

 public:
  qt_compute_helper(size_t sz, size_t window_sz, bool dp,
                    SCAMPArchitecture arch)
      : size(sz), window_size(window_sz), double_precision(dp), _arch(arch) {
    init();
  }

#ifdef _HAS_CUDA_

  ~qt_compute_helper() { free(); }

  SCAMPError_t compute_QT(double *QT, const double *T, const double *Q,
                          const double *qmeans, cudaStream_t s);

#endif

  SCAMPError_t compute_QT_CPU(double *QT, const double *T, const double *Q);
};

}  // namespace SCAMP
