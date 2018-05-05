#pragma once

#include <cufft.h>
#include "common.h"

namespace SCRIMP {

class fft_precompute_helper {
private:
    double *Q_reverse_pad;
    cuDoubleComplex *Qc, *Tc;
    cufftHandle fft_plan, ifft_plan;    
    const size_t size;
    const size_t window_size;
    const size_t cufft_data_size;
    const bool double_precision;
    const int fft_work_size = 512;    
public:
    fft_precompute_helper(size_t sz, size_t window_sz, bool dp) : size(sz), window_size(window_sz), cufft_data_size(sz / 2 + 1), double_precision(dp) {
        if(double_precision) {
            CHECK_CUFFT_ERRORS(cufftPlan1d(&fft_plan, size, CUFFT_D2Z, 1));
            CHECK_CUFFT_ERRORS(cufftPlan1d(&ifft_plan, size, CUFFT_Z2D, 1));
        } else {
            CHECK_CUFFT_ERRORS(cufftPlan1d(&fft_plan, size, CUFFT_R2C, 1));
            CHECK_CUFFT_ERRORS(cufftPlan1d(&ifft_plan, size, CUFFT_C2R, 1));
        }
        cudaMalloc(&Q_reverse_pad, sizeof(double) * size);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&Tc, sizeof(cuDoubleComplex) * cufft_data_size);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&Qc, sizeof(cuDoubleComplex) * cufft_data_size);
        gpuErrchk(cudaPeekAtLastError());
    }
    ~fft_precompute_helper() {
        cudaFree(Q_reverse_pad);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(Tc);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(Qc);
        gpuErrchk(cudaPeekAtLastError());
        CHECK_CUFFT_ERRORS(cufftDestroy(fft_plan));
        CHECK_CUFFT_ERRORS(cufftDestroy(ifft_plan));
    }
    SCRIMPError_t compute_QT(double *QT_scratch, const double *timeseries, const double *query, const double *qmeans, cudaStream_t s);

};



}
