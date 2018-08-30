#include "fft_helper.h"
#include <cufft.h> 

namespace SCAMP {

__global__ void elementwise_multiply_inplace(const cuDoubleComplex* A, cuDoubleComplex* B, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
       B[tid] = cuCmul(A[tid], B[tid]);
    }
}

// A is input unaligned sliding dot products produced by ifft
// out is the computed vector of distances
__global__ void normalized_aligned_dot_products(const double* A, const double divisor,
                                                const unsigned int m, const unsigned int n,
                                                double* QT)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < n) {
        QT[a] = A[a + m - 1] / divisor;
    }
}

__global__ void populate_reverse_pad(const double *Q, double *Q_reverse_pad, const double *mean, const int window_size, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double mu = *mean;
    if(tid < window_size) {
        Q_reverse_pad[tid] = Q[window_size - 1 - tid] - mu;
    }else if(tid < size){ 
        Q_reverse_pad[tid] = 0;
    }
}

SCAMPError_t fft_precompute_helper::compute_QT(double* QT, const double* T, const double *Q, const double *qmeans, cudaStream_t s)
{        

    cufftResult cufftError;
    cudaError_t error;

    const int n = size - window_size + 1;
    dim3 block(fft_work_size, 1, 1);
    

    cufftError = cufftSetStream(fft_plan, s);
    if (cufftError != CUFFT_SUCCESS) {
        return SCAMP_CUFFT_ERROR;
    }
    cufftError = cufftSetStream(ifft_plan,s);
    if (cufftError != CUFFT_SUCCESS) {
        return SCAMP_CUFFT_ERROR;
    }
    
    // Compute the FFT of the time series
    cufftError = cufftExecD2Z(fft_plan, const_cast<double*>(T), Tc);

    if (cufftError != CUFFT_SUCCESS) {
        return SCAMP_CUFFT_EXEC_ERROR;
    }
    
    // Reverse and zero pad the query
    populate_reverse_pad<<<dim3(ceil(size / (float) fft_work_size),1,1), block, 0, s>>>(Q, Q_reverse_pad, qmeans, window_size, size);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        return SCAMP_CUDA_ERROR;
    }
    
    cufftError = cufftExecD2Z(fft_plan, Q_reverse_pad, Qc);
    if (cufftError != CUFFT_SUCCESS) {
        return SCAMP_CUFFT_EXEC_ERROR;
    }
    
    elementwise_multiply_inplace<<<dim3(ceil(cufft_data_size / (float) fft_work_size), 1, 1), block, 0, s>>>(Tc, Qc, cufft_data_size);
    error = cudaPeekAtLastError();
    if ( error != cudaSuccess) {
        return SCAMP_CUDA_ERROR;
    }

    cufftError = cufftExecZ2D(ifft_plan, Qc, Q_reverse_pad);

    if (cufftError != CUFFT_SUCCESS) {
        return SCAMP_CUFFT_EXEC_ERROR;
    }
    
    normalized_aligned_dot_products<<<dim3(ceil(n / (float) fft_work_size), 1, 1), block, 0, s>>>(Q_reverse_pad, size, window_size, n, QT);
    error = cudaPeekAtLastError();

    if(error != cudaSuccess) {
        return SCAMP_CUDA_ERROR;
    }

    return SCAMP_NO_ERROR;
    
}

} 
