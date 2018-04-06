#pragma once
#include "common.h"
#include <float.h>

namespace SCRIMP {


//Atomically updates the MP/idxs using a single 64-bit integer. We lose a small amount of precision in the output, if we do not do this we are unable
// to atomically update both the matrix profile and the indexes without using a critical section and dedicated locks.
__device__ inline void MPatomicMax(volatile unsigned long long int* address, float val, unsigned int idx)
{
    mp_entry loc, loctest;
    loc.floats[0] = val;
    loc.ints[1] = idx;
    loctest.ulong = *address;
    while (loctest.floats[0] < val){
        loctest.ulong = atomicCAS((unsigned long long int*) address, loctest.ulong,  loc.ulong);
    }
}

template<class DTYPE, unsigned int BLOCKSZ, unsigned int tile_height>
__device__ inline void initialize_tile_memory(const unsigned long long int *profile_A, const unsigned long long int *profile_B,
                                              const double *T_a, const double *T_b, const double *means_A, const double *means_B,
                                              const double *inv_stds_A, const double *inv_stds_B,
                                              volatile mp_entry localMPMain[], volatile mp_entry localMPOther[],
                                              DTYPE A_low[], DTYPE A_high[], DTYPE B_low[], DTYPE B_high[],
                                              DTYPE mean_x[], DTYPE mean_y[], DTYPE inv_std_x[],
                                              DTYPE inv_std_y[], const unsigned int n, const unsigned int m,
                                              const unsigned int mainStart, const unsigned int otherStart,
                                              const unsigned int x, const unsigned int y)
{
    // Update local cache to point to the next chunk of the MP
    // We may not get the 'freshest' values from the global array, but it doesn't really matter too much
    if (mainStart + threadIdx.x < n) {
        localMPMain[threadIdx.x].ulong = profile_A[mainStart + threadIdx.x];
    } else {
        localMPMain[threadIdx.x].floats[0] = CC_MIN;
        localMPMain[threadIdx.x].ints[1] = 0;
    }

    // Each thread grabs 2 values for the main cache
    if (threadIdx.x < tile_height && mainStart+threadIdx.x+BLOCKSZ < n) {
        localMPMain[BLOCKSZ + threadIdx.x].ulong = profile_A[mainStart + BLOCKSZ + threadIdx.x];
    } else if (threadIdx.x < tile_height) {
        localMPMain[threadIdx.x + BLOCKSZ].floats[0] = CC_MIN;
        localMPMain[threadIdx.x + BLOCKSZ].ints[1] = 0;
    }
    
    // We also update the cache for the transposed tile
    if (threadIdx.x < tile_height && otherStart+threadIdx.x < n) {
        localMPOther[threadIdx.x].ulong = profile_B[otherStart + threadIdx.x];
    } else if (threadIdx.x < tile_height) {
        localMPOther[threadIdx.x].floats[0] = CC_MIN;
        localMPOther[threadIdx.x].ints[1] = 0;
    }

    // Update the other cached values to reflect the upcoming tile
    if (x <  n + m - 1) {
        A_low[threadIdx.x] = T_a[x];
    }
    if (threadIdx.x < tile_height && x + BLOCKSZ < n + m - 1) {
        A_low[threadIdx.x + BLOCKSZ] = T_a[x + BLOCKSZ];
    }
    
    if (x + m < n + m - 1) {
        A_high[threadIdx.x] = T_a[x + m];
    }
    if (threadIdx.x < tile_height && x + BLOCKSZ + m < n + m - 1) {
        A_high[threadIdx.x + BLOCKSZ] = T_a[x + BLOCKSZ + m];
    }
    if (threadIdx.x < tile_height && y + threadIdx.x < n + m - 1) {
        B_low[threadIdx.x] = T_b[y + threadIdx.x];
    }
    if (threadIdx.x < tile_height && y + threadIdx.x + m < n + m - 1) {
        B_high[threadIdx.x] = T_b[y + threadIdx.x + m];
    }
    if (x < n) {
        inv_std_x[threadIdx.x] = inv_stds_A[x];
        // We precompute part of the distance calculation in the mean_x variable
        // This saves us a multiply in the main loop
        mean_x[threadIdx.x] = means_A[x] * m;
    }
    if (threadIdx.x < tile_height && x + BLOCKSZ < n) {
        inv_std_x[threadIdx.x + BLOCKSZ] = inv_stds_A[x + BLOCKSZ];
        // We precompute part of the distance calculation in the mean_x variable
        // This saves us a multiply in the main loop
        mean_x[threadIdx.x + BLOCKSZ] = means_A[x + BLOCKSZ] * m;
    }
    if (threadIdx.x < tile_height && y + threadIdx.x < n) {
        inv_std_y[threadIdx.x] = inv_stds_B[y + threadIdx.x];
        mean_y[threadIdx.x] = means_B[y + threadIdx.x];
    }
}

//Computes the matrix profile given the sliding dot products for the first query and the precomputed data statisics
template<class DTYPE_IN, class DTYPE, unsigned int BLOCKSZ, unsigned int UNROLL_COUNT>
__global__ void do_tile_self_join_upper(const DTYPE_IN* QT, const DTYPE_IN *T_a, const DTYPE_IN *T_b, const DTYPE_IN *inv_stds_A, const DTYPE_IN *inv_stds_B, const DTYPE_IN *means_A,
                                        const DTYPE_IN *means_B, unsigned long long int *profile_A, unsigned long long int *profile_B, unsigned int m, unsigned int n, unsigned int global_start_x, unsigned int global_start_y,
                                        struct reg_mem<UNROLL_COUNT> mem)
{
    // Factor and threads per block must both be powers of two where: factor <= threads per block
    // UNROLL_COUNT * factor must also evenly divide WORK_SIZE
    // 'factor' is a scaling factor for the tile size, due to shared memory considerations
    // we cannot do a full tile at once, we must chop it into pieces
    // The values that are set here should give good performance already
    // but may be fine tuned for your specific Nvidia architecture
    const int factor = 2;
    const int tile_height = BLOCKSZ / factor;
    const int tile_width = tile_height + BLOCKSZ;
    __shared__ mp_entry localMPMain[tile_width];
    __shared__ mp_entry localMPOther[tile_height];
    __shared__ DTYPE A_low[tile_width];
    __shared__ DTYPE A_high[tile_width];
    __shared__ DTYPE inv_std_x[tile_width];
    __shared__ DTYPE inv_std_y[tile_height];
    __shared__ DTYPE mean_x[tile_width];
    __shared__ DTYPE mean_y[tile_height];
    __shared__ DTYPE B_high[tile_height];
    __shared__ DTYPE B_low[tile_height];

    // This is the index of the meta-diagonal that this thread block will work on
    int meta_diagonal_idx = blockIdx.x;

    // The first threads are acutally computing the trivial match between the same subsequence
    // we exclude these from the calculation
    const int exclusion = (m / 4);
    int tile_start_x = meta_diagonal_idx * BLOCKSZ;
    int tile_start_y = 0;
    
    // x is the global column of the distance matrix
    // y is the global row of the distance matrix
    // localX, localY are the local coordinates of the thread position in the tile it is working on
    int x = tile_start_x + threadIdx.x;
    int y = 0;

    bool excluded;
    if(global_start_x + x >= global_start_y && global_start_x + x <= global_start_y + exclusion) {
        excluded = true;
    }else {
        excluded = false;
    }

    int localX, localY;

    // Load the first dot product value
    if (x < n) {
        mem.qt[0] = QT[x];
    }

    /////////////////////////////////////    
    // Main loop
    /////////////////////////////////////
    // Each threadblock finds all the distances on a 'metadiagonal'
    // We use a tiled approach for each thread block
    // The tiles are horizontal slices of the diagonal, think of a parallelogram cut
    // from a diagonal slice of the distance matrix 
    // Each thread starts on the first row and works its way down-right towards right
    // side of the distance matrix
    while (tile_start_x < n)
    {
        // Initialize the next tile's shared memory
        initialize_tile_memory<DTYPE, BLOCKSZ, tile_height>(profile_A, profile_B, T_a, T_b, means_A, means_B, inv_stds_A, inv_stds_B, localMPMain, localMPOther,
                                                A_low, A_high, B_low, B_high, mean_x, mean_y, inv_std_x,
                                                inv_std_y, n, m, tile_start_x, tile_start_y, x, y);

        // Reset the tile local positions
        localY = 0;
        localX = threadIdx.x;

        // Start of new tile, sync
        __syncthreads();


        // Process the tile:
        // Each iteration generates the next UNROLL_COUNT distances
        // This loop is partially unrolled to improve instruction level parallelism
        // In all but the last tile in each metadiagonal, this first loop will compute
        // the entire tile, at the end we will have some leftover (UNROLL_COUNT may
        // not cleanly divide x) which is handled by the second loop
        if(!excluded) {
            while (x < n - UNROLL_COUNT + 1 && localY < tile_height)
            {
                // Update the QT value for the next iteration(s)
                #pragma unroll
                for (int i = 0; i < UNROLL_COUNT - 1; ++i) {
                    mem.qt[i + 1] = mem.qt[i] - A_low[localX + i] * B_low[localY + i] + A_high[localX + i] * B_high[localY + i];
                }

                // Compute the next partial distance value(s):
                // We defer some of the calculation until after the kernel has finished, this saves us several
                // long latency math operations in this critical path.
                // The distance computed here can be converted to the true z-normalized euclidan
                // distance in constant time
                // mean_x has already been multiplied with the window size 'm' when the tile was populated
                // This saves us an extra multiply for each distance computed
                #pragma unroll
                for (int i = 0; i < UNROLL_COUNT; ++i) {
                    mem.dist[i] = (static_cast<float>(mem.qt[i]) - (mean_x[localX + i] * mean_y[localY + i])) * inv_std_x[localX + i] * inv_std_y[localY + i];
                }

                // This is the next qt value that will be used in the next iteration of the loop
                mem.qt[0] = mem.qt[UNROLL_COUNT - 1] - A_low[localX + UNROLL_COUNT - 1] * B_low[localY + UNROLL_COUNT - 1] + A_high[localX + UNROLL_COUNT - 1] * B_high[localY + UNROLL_COUNT - 1];

                // Update the cache with the new max value atomically
                // This is a major source of latency, but this is probably still the best option
                // if you can think of a better way to handle this please let me know
                #pragma unroll
                for (int i = 0; i < UNROLL_COUNT; ++i) {
                    MPatomicMax((unsigned long long*) (localMPMain + localX + i), mem.dist[i], global_start_y + y + i);
                    MPatomicMax((unsigned long long*) (localMPOther + localY + i), mem.dist[i], global_start_x + x + i);
                }

                x += UNROLL_COUNT;
                y += UNROLL_COUNT;
                localX += UNROLL_COUNT;
                localY += UNROLL_COUNT;
            }
            double qt_curr = mem.qt[0];

        
            // Finish the remaining iterations of the final tile if there were leftover
            // NOTE: this loop should only execute once for each thread beacuse we restrict
            // UNROLL_COUNT to be a factor of tile_height
            while (x < n && localY < tile_height) {
                float dist = (static_cast<float>(qt_curr) - (mean_x[localX] * mean_y[localY])) * inv_std_x[localX] * inv_std_y[localY];
                qt_curr = qt_curr - A_low[localX] * B_low[localY] + A_high[localX] * B_high[localY];
                MPatomicMax((unsigned long long*) (localMPMain + localX), dist, global_start_y + y);
                MPatomicMax((unsigned long long*) (localMPOther + localY), dist, global_start_x + x);
                x++;
                y++;
                localX++;
                localY++;
            }
        } else {
            x += tile_height;
            y += tile_height;
        }
        // After this sync, the caches will be updated with the best so far values for this tile
        __syncthreads();

        // If we updated any values in the cached MP, try to push them to the global "master" MP
        if (tile_start_x + threadIdx.x < n) {
            MPatomicMax(profile_A + tile_start_x + threadIdx.x, localMPMain[threadIdx.x].floats[0], localMPMain[threadIdx.x].ints[1]);
        }
        if (tile_start_x + threadIdx.x + BLOCKSZ < n && threadIdx.x < tile_height) {
            MPatomicMax(profile_A + BLOCKSZ + tile_start_x + threadIdx.x, localMPMain[threadIdx.x + BLOCKSZ].floats[0], localMPMain[threadIdx.x + BLOCKSZ].ints[1]);
        }
        if (tile_start_y + threadIdx.x < n && threadIdx.x < tile_height) {
            MPatomicMax(profile_B + tile_start_y + threadIdx.x, localMPOther[threadIdx.x].floats[0], localMPOther[threadIdx.x].ints[1]);
        }

        // Update the tile position
        tile_start_x += tile_height;
        tile_start_y += tile_height;

        // Make sure our updates were committed before we pull in the next tile
        __threadfence_block();
    }
    



}

template <class DATATYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t kernel_self_join_upper(const DATATYPE *QT, const DATATYPE *timeseries_A, const DATATYPE *timeseries_B, const DATATYPE *std_dev_A,
                                     const DATATYPE *std_dev_B, const DATATYPE *means_A, const DATATYPE *means_B, unsigned long long int *profile_A,
                                     unsigned long long int *profile_B, size_t window_size, size_t tile_width, size_t global_x, size_t global_y, cudaStream_t s)
{
        do_tile_self_join_upper<DATATYPE, float, BLOCKSZ, UNROLL_COUNT><<<dim3(ceil(tile_width / (double) BLOCKSZ), 1, 1), dim3(BLOCKSZ, 1,1), 0, s>>>(QT, timeseries_A, timeseries_B, std_dev_A, std_dev_B, means_A, means_B, profile_A, profile_B, window_size, tile_width, global_x, global_y, reg_mem<UNROLL_COUNT>());
        cudaError_t err = cudaPeekAtLastError();
        if(err != cudaSuccess) {
            return SCRIMP_CUDA_ERROR;
        }
        return SCRIMP_NO_ERROR;

}


template <class DATATYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t kernel_self_join_lower(const DATATYPE *QT, const DATATYPE *timeseries_A, const DATATYPE *timeseries_B, const DATATYPE *std_dev_A,
                                     const DATATYPE *std_dev_B, const DATATYPE *means_A, const DATATYPE *means_B, unsigned long long int *profile_A,
                                     unsigned long long int *profile_B, size_t window_size, size_t tile_width, size_t tile_height, size_t global_x, size_t global_y, cudaStream_t s)
{
        
        //do_tile_self_join_upper<DATATYPE, float, BLOCKSZ, UNROLL_COUNT><<<dim3(ceil(tile_width / (double) BLOCKSZ), 1, 1), dim3(BLOCKSZ, 1,1), 0, s>>>(QT, timeseries_A, timeseries_B, means_A, means_B, std_dev_A, std_dev_B, profile_A, profile_B, profile_A_sym, profile_B_sym, window_size, tile_width, reg_mem<UNROLL_COUNT>());
        return SCRIMP_NO_ERROR;
}



}
