#pragma once

#include <cufft.h>
#include "common.h"

namespace SCRIMP {

template <class DATATYPE, class CUFFT_DTYPE>
class fft_precompute_helper {
private:
    DATATYPE *Q_reverse_pad;
    CUFFT_DTYPE *Qc, *Tc;
    cufftHandle fft_plan, ifft_plan;    
    const size_t size;
    const size_t window_size;
    const size_t cufft_data_size;
    const bool double_precision;
        
public:
    fft_precompute_helper<DATATYPE, CUFFT_DTYPE>(size_t sz, size_t window_sz, bool dp) : size(sz), window_size(window_sz), cufft_data_size(sz / 2 + 1), double_precision(dp) {
        if(double_precision) {
            CHECK_CUFFT_ERRORS(cufftPlan1d(&fft_plan, size, CUFFT_D2Z, 1));
            CHECK_CUFFT_ERRORS(cufftPlan1d(&ifft_plan, size, CUFFT_Z2D, 1));
        } else {
            CHECK_CUFFT_ERRORS(cufftPlan1d(&fft_plan, size, CUFFT_R2C, 1));
            CHECK_CUFFT_ERRORS(cufftPlan1d(&ifft_plan, size, CUFFT_C2R, 1));
        }
        cudaMalloc(&Q_reverse_pad, sizeof(DATATYPE) * size);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&Tc, sizeof(CUFFT_DTYPE) * cufft_data_size);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&Qc, sizeof(CUFFT_DTYPE) * cufft_data_size);
        gpuErrchk(cudaPeekAtLastError());
    }
    ~fft_precompute_helper<DATATYPE, CUFFT_DTYPE>() {
        cudaFree(Q_reverse_pad);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(Tc);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(Qc);
        gpuErrchk(cudaPeekAtLastError());
        CHECK_CUFFT_ERRORS(cufftDestroy(fft_plan));
        CHECK_CUFFT_ERRORS(cufftDestroy(ifft_plan));
    }
    SCRIMPError_t compute_QT(DATATYPE *QT_scratch, const DATATYPE *timeseries, const DATATYPE *query, cudaStream_t s);

};


#define WORK_SIZE 512
template<class DTYPE>
__global__ void elementwise_multiply_inplace(const DTYPE* A, DTYPE *B, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
       B[tid] *= A[tid];
    }
} 

template<>
__global__ void elementwise_multiply_inplace(const cuDoubleComplex* A, cuDoubleComplex* B, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
       B[tid] = cuCmul(A[tid], B[tid]);
    }
}

template<>
__global__ void elementwise_multiply_inplace(const cuComplex* A, cuComplex* B, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
       B[tid] = cuCmulf(A[tid], B[tid]);
    }
}

// A is input unaligned sliding dot products produced by ifft
// out is the computed vector of distances
template<class DTYPE>
__global__ void normalized_aligned_dot_products(const DTYPE* A, const DTYPE divisor,
                                                const unsigned int m, const unsigned int n,
                                                DTYPE* QT)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < n) {
        QT[a] = A[a + m - 1] / divisor;
    }
}

template<class DTYPE>
__global__ void populate_reverse_pad(const DTYPE *Q, DTYPE *Q_reverse_pad, const int window_size, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < window_size) {
        Q_reverse_pad[tid] = Q[window_size - 1 - tid];
    }else if(tid < size){ 
        Q_reverse_pad[tid] = 0;
    }
}

template<class DATATYPE, class CUFFT_DTYPE>
SCRIMPError_t fft_precompute_helper<DATATYPE, CUFFT_DTYPE>::compute_QT(DATATYPE* QT, const DATATYPE* T, const DATATYPE *Q, cudaStream_t s)
{        

    cufftResult cufftError;
    cudaError_t error;

    const int n = size - window_size + 1;
    dim3 block(WORK_SIZE, 1, 1);
    

    cufftError = cufftSetStream(fft_plan, s);
    if (cufftError != CUFFT_SUCCESS) {
        return SCRIMP_CUFFT_ERROR;
    }
    cufftError = cufftSetStream(ifft_plan,s);
    if (cufftError != CUFFT_SUCCESS) {
        return SCRIMP_CUFFT_ERROR;
    }
    
    // Compute the FFT of the time series
//    if (double_precision) {
        cufftError = cufftExecD2Z(fft_plan, const_cast<DATATYPE*>(T), Tc);
//    } else {
//        cufftError = cufftExecR2C(fft_plan, const_cast<float*>(T), Tc);
//    }

    if (cufftError != CUFFT_SUCCESS) {
        return SCRIMP_CUFFT_EXEC_ERROR;
    }
    
    // Reverse and zero pad the query
    populate_reverse_pad<DATATYPE><<<dim3(ceil(size / (float) WORK_SIZE),1,1), block, 0, s>>>(Q, Q_reverse_pad, window_size, size);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        return SCRIMP_CUDA_ERROR;
    }
    
    // Compute the FFT of the query
//    if (double_precision) {
        cufftError = cufftExecD2Z(fft_plan, Q_reverse_pad, Qc);
//    } else {
//        cufftError = cufftExecR2C(fft_plan, Q_reverse_pad, Qc);
//    }
    if (cufftError != CUFFT_SUCCESS) {
        return SCRIMP_CUFFT_EXEC_ERROR;
    }
    
    elementwise_multiply_inplace<CUFFT_DTYPE><<<dim3(ceil(cufft_data_size / (float) WORK_SIZE), 1, 1), block, 0, s>>>(Tc, Qc, cufft_data_size);
    error = cudaPeekAtLastError();
    if ( error != cudaSuccess) {
        return SCRIMP_CUDA_ERROR;
    }

    // Compute the ifft
    // Use the space for the query as scratch space as we no longer need it
//    if (double_precision) {
        cufftError = cufftExecZ2D(ifft_plan, Qc, Q_reverse_pad);
//    } else {
//        cufftError = cufftExecC2R(ifft_plan, Qc, Q_reverse_pad);
//    }

    if (cufftError != CUFFT_SUCCESS) {
        return SCRIMP_CUFFT_EXEC_ERROR;
    }
    
    normalized_aligned_dot_products<DATATYPE><<<dim3(ceil(n / (float) WORK_SIZE), 1, 1), block, 0, s>>>(Q_reverse_pad, size, window_size, n, QT);
    error = cudaPeekAtLastError();

    if(error != cudaSuccess) {
        return SCRIMP_CUDA_ERROR;
    }

    return SCRIMP_NO_ERROR;
    
}

}
