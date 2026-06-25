// CUDA-to-HIP compatibility header for SCAMP
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// On ROCm/HIP builds this aliases the CUDA spellings used throughout SCAMP
// to their HIP equivalents. On NVIDIA builds it is a no-op pass-through to
// the CUDA runtime.
//
// Warp-size abstraction:
//   Device code: kWarpSize is 64 on CDNA (gfx90a, gfx94x), 32 elsewhere.
//   Host code: query hipGetDeviceProperties(&prop, dev).warpSize at runtime.
//   Shuffle masks: HIP requires 64-bit masks regardless of wave width.

#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

// hipCUB pulls in rocPRIM device intrinsics that only compile under the HIP
// device compiler; it is consumed solely by the DeviceMergeSort call in
// kernels.cu. Keeping it out of host .cpp translation units (which reach this
// header via kernels.h) prevents those intrinsics from leaking into g++ host
// compilation.
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#include <hipcub/hipcub.hpp>
#endif

// ----------------------------------------------------------------------------
// Warp-size abstraction
// ----------------------------------------------------------------------------
// Device code: per-arch constant. __GFX9__ is defined only during device
// compilation for CDNA (gfx90a, gfx94x).
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__GFX9__)
static constexpr int kWarpSize = 64;  // CDNA: gfx90a, gfx94x
#else
static constexpr int kWarpSize = 32;  // RDNA: gfx10xx, gfx11xx, gfx12xx
#endif
#else
// Host code: use runtime query hipGetDeviceProperties().warpSize
static constexpr int kWarpSize = 64;  // Compile-time upper bound for arrays
#endif

// Upper bound for static shared-memory arrays sized by warp count.
static constexpr int kWarpSizeUpperBound = 64;

// Full-warp shuffle mask. HIP requires 64-bit masks regardless of wave width.
#define SCAMP_FULL_WARP_MASK 0xffffffffffffffffULL

// ----------------------------------------------------------------------------
// CUDA runtime -> HIP runtime
// ----------------------------------------------------------------------------
#define cudaMalloc             hipMalloc
#define cudaFree               hipFree
#define cudaMallocAsync        hipMallocAsync
#define cudaFreeAsync          hipFreeAsync
#define cudaMemcpy             hipMemcpy
#define cudaMemcpyAsync        hipMemcpyAsync
#define cudaMemset             hipMemset
#define cudaMemsetAsync        hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define cudaStream_t           hipStream_t
#define cudaStreamCreate       hipStreamCreate
#define cudaStreamDestroy      hipStreamDestroy
#define cudaStreamSynchronize  hipStreamSynchronize

#define cudaEvent_t            hipEvent_t
#define cudaEventCreate        hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy       hipEventDestroy
#define cudaEventRecord        hipEventRecord
#define cudaEventSynchronize   hipEventSynchronize
#define cudaEventElapsedTime   hipEventElapsedTime

#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess
#define cudaGetLastError       hipGetLastError
#define cudaGetErrorString     hipGetErrorString
#define cudaPeekAtLastError    hipPeekAtLastError
#define cudaGetDevice          hipGetDevice
#define cudaSetDevice          hipSetDevice
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp         hipDeviceProp_t

#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrWarpSize    hipDeviceAttributeWarpSize

#define cudaFuncSetAttribute   hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// ----------------------------------------------------------------------------
// cuFFT -> hipFFT
// ----------------------------------------------------------------------------
#define cufftHandle            hipfftHandle
#define cufftResult            hipfftResult
#define cufftResult_t          hipfftResult_t

#define CUFFT_SUCCESS          HIPFFT_SUCCESS
#define CUFFT_INVALID_PLAN     HIPFFT_INVALID_PLAN
#define CUFFT_ALLOC_FAILED     HIPFFT_ALLOC_FAILED
#define CUFFT_INVALID_TYPE     HIPFFT_INVALID_TYPE
#define CUFFT_INVALID_VALUE    HIPFFT_INVALID_VALUE
#define CUFFT_INTERNAL_ERROR   HIPFFT_INTERNAL_ERROR
#define CUFFT_EXEC_FAILED      HIPFFT_EXEC_FAILED
#define CUFFT_SETUP_FAILED     HIPFFT_SETUP_FAILED
#define CUFFT_INVALID_SIZE     HIPFFT_INVALID_SIZE
#define CUFFT_UNALIGNED_DATA   HIPFFT_UNALIGNED_DATA
#define CUFFT_INCOMPLETE_PARAMETER_LIST HIPFFT_INCOMPLETE_PARAMETER_LIST
#define CUFFT_INVALID_DEVICE   HIPFFT_INVALID_DEVICE
#define CUFFT_PARSE_ERROR      HIPFFT_PARSE_ERROR
#define CUFFT_NO_WORKSPACE     HIPFFT_NO_WORKSPACE
#define CUFFT_NOT_IMPLEMENTED  HIPFFT_NOT_IMPLEMENTED
#define CUFFT_NOT_SUPPORTED    HIPFFT_NOT_SUPPORTED

#define CUFFT_D2Z              HIPFFT_D2Z
#define CUFFT_Z2D              HIPFFT_Z2D
#define CUFFT_R2C              HIPFFT_R2C
#define CUFFT_C2R              HIPFFT_C2R

#define cufftPlan1d            hipfftPlan1d
#define cufftDestroy           hipfftDestroy
#define cufftSetStream         hipfftSetStream
#define cufftExecD2Z           hipfftExecD2Z
#define cufftExecZ2D           hipfftExecZ2D
#define cufftExecR2C           hipfftExecR2C
#define cufftExecC2R           hipfftExecC2R

// Complex types and arithmetic
#define cuDoubleComplex        hipDoubleComplex
#define cuFloatComplex         hipFloatComplex
#define cuComplex              hipComplex
#define cuCmul                 hipCmul

// ----------------------------------------------------------------------------
// CUB -> hipCUB
// ----------------------------------------------------------------------------
#define cub                    hipcub

// ----------------------------------------------------------------------------
// Warp-level intrinsics (shuffle masks must be 64-bit on HIP)
// ----------------------------------------------------------------------------
// The upstream SCAMP code uses 0xffffffff (32-bit) masks. On HIP these must
// be 64-bit. We provide wrapper macros so call sites stay unchanged. The
// actual SCAMP code paths that use shuffles are patched to use
// SCAMP_FULL_WARP_MASK directly for clarity.

#else  // CUDA path

#include <cuda_runtime.h>
#include <cufft.h>

// CUB's DeviceMergeSort headers carry device intrinsics that only compile
// under nvcc device passes; they are consumed solely by the DeviceMergeSort
// call in kernels.cu. Gating them on __CUDACC__ mirrors the HIP-side hipCUB
// guard and keeps them out of g++ host translation units (tile.cpp,
// qt_helper.cpp) that reach this header via kernels.h.
#if defined(__CUDACC__)
#include <cub/device/device_merge_sort.cuh>
#endif

// On CUDA, kWarpSize is always 32.
static constexpr int kWarpSize = 32;
static constexpr int kWarpSizeUpperBound = 32;

// Full-warp shuffle mask for CUDA (32-bit).
#define SCAMP_FULL_WARP_MASK 0xffffffff

#endif  // USE_HIP
