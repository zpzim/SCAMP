#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/transform_scan.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#define CC_MIN -FLT_MAX

namespace SCRIMP {


typedef union  {
  float floats[2];                 // floats[0] = lowest
  unsigned int ints[2];                     // ints[1] = lowIdx
  unsigned long long int ulong;    // for atomic update
} mp_entry;

template<unsigned int count>
struct reg_mem {
    float dist[count];
    double qt[count];
};

enum SCRIMPError_t { SCRIMP_NO_ERROR, SCRIMP_FUNCTIONALITY_UNIMPLEMENTED, SCRIMP_TILE_ILLEGAL_TYPE, SCRIMP_CUDA_ERROR, SCRIMP_CUFFT_ERROR, SCRIMP_CUFFT_EXEC_ERROR, SCRIMP_DIM_INCOMPATIBLE };

enum SCRIMPTileType { SELF_JOIN_FULL_TILE, SELF_JOIN_UPPER_TRIANGULAR, AB_JOIN_FULL_TILE, AB_FULL_JOIN_FULL_TILE };

struct MPIDXCombine
{
	__host__ __device__
	unsigned long long int operator()(double x, unsigned int idx){
		mp_entry item;
		item.floats[0] = (float) x;
		item.ints[1] = idx;
		return item.ulong;
	}
};

//Returns the maximum between 2 matrix profile candidates, also records the index of the minumum value
//For constructing the final matrix profile
struct max_with_index
{

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{

		// A[i] = min(A[i], B[i]);
		// C[i] = A[i] == min(A[i], B[i]) ? i : C[i];
		mp_entry other;
		other.ulong = thrust::get<2>(t);
        if(thrust::get<0>(t) < other.floats[0]){
			thrust::get<0>(t) = other.floats[0];
			thrust::get<1>(t) = other.ints[1];
		}

	}
};

}

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
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

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}
#endif

#define CHECK_CUFFT_ERRORS(call) { \
    cufftResult_t err; \
    if ((err = (call)) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _cudaGetErrorEnum(err), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}

//This macro checks return value of the CUDA runtime call and exits
//the application if the call failed.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


