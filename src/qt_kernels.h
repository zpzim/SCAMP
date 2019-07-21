#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

void launch_elementwise_multiply_inplace(const cuDoubleComplex *A,
                                         cuDoubleComplex *B, const int size,
                                         int fft_work_size, cudaStream_t s);
void launch_populate_reverse_pad(const double *Q, double *Q_reverse_pad,
                                 const double *mean, const int window_size,
                                 const int size, int fft_work_size,
                                 cudaStream_t s);
void launch_normalized_aligned_dot_products(const double *A,
                                            const double divisor,
                                            const unsigned int m,
                                            const unsigned int n, double *QT,
                                            int fft_work_size, cudaStream_t s);
