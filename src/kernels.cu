#include "kernels.h"

namespace SCRIMP {

#define BLOCKSZ 512
#define BLOCKSPERSM 2
#define TILE_HEIGHT 200

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

// As above, but checks a previously read value before attempting another read
// This allows us to exploit vectorized loads of the matrix profile
__device__ inline void MPatomicMax_check(volatile unsigned long long int* __restrict__ address, float val, unsigned int idx, float curr_val)
{
    if(val > curr_val) {
        mp_entry loc, loctest;
        loc.floats[0] = val;
        loc.ints[1] = idx;
        loctest.ulong = *address;
        while (loctest.floats[0] < val){
            loctest.ulong = atomicCAS((unsigned long long int*) address, loctest.ulong,  loc.ulong);
        }
     }
}

__device__ inline void MPMax(const float d1, const float d2, const unsigned int i1,
                             const unsigned int i2, float &outd, unsigned int &outi)
{
    if(d1 >= d2) {
        outd = d1;
        outi = i1;
    } else {
        outd = d2;
        outi = i2;
    }

}

// Computes max(a,b) with index and stores the result in a
__device__ inline void MPMax2(float &d1, const float &d2, unsigned int &i1,
                             const unsigned int &i2)
{
    if(d2 > d1) {
        d1 = d2;
        i1 = i2;
    } 
}


// Computes the max of 4 values in a float 4
__device__ inline float max4(const float4 &d, const unsigned int init, unsigned int &idx) {
    float ret = d.x;
    idx = init;
    if(d.y > ret) {
        ret = d.y;
        idx = init + 1;
    }
    if(d.z > ret) {
        ret = d.z;
        idx = init + 2;
    }
    if(d.w > ret) {
        ret = d.w;
        idx = init + 3;
    }
    return ret;
}

template<int tile_height, int tile_width>
__device__  inline void initialize_tile_memory(const unsigned long long int* __restrict__ profile_A,
                                        const unsigned long long int* __restrict__ profile_B,
                                        const double* __restrict__ df_A, const double* __restrict__ df_B,
                                        const double* __restrict__ dg_A, const double* __restrict__ dg_B,
                                        const double* __restrict__ norms_A, const double* __restrict__ norms_B,
                                        mp_entry* __restrict__ local_mp_col, mp_entry* __restrict__ local_mp_row,
                                        float* __restrict__ df_col, float* __restrict__ df_row, float* __restrict__ dg_col,
                                        float* __restrict__ dg_row, float* __restrict__ norm_col, float* __restrict__ norm_row,
                                        const unsigned int n_x, const unsigned int n_y, const unsigned int col_start,
                                        const unsigned int row_start)
{
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while(local_position < tile_width && global_position < n_x) {
        dg_col[local_position] = dg_A[global_position];
        df_col[local_position] = df_A[global_position];
        norm_col[local_position] = norms_A[global_position];
        local_mp_col[local_position].ulong = profile_A[global_position];
        local_position += BLOCKSZ;
        global_position += BLOCKSZ;
    }

    global_position = row_start + threadIdx.x;
    local_position = threadIdx.x;
    while(local_position < tile_height && global_position < n_y) {
        dg_row[local_position] = dg_B[global_position];
        df_row[local_position] = df_B[global_position];
        norm_row[local_position] = norms_B[global_position];
        local_mp_row[local_position].ulong = profile_B[global_position];
        local_position += BLOCKSZ;
        global_position += BLOCKSZ;
    
    }

}

// This does one row of work for 4 diagonals in a single thread
__device__ inline void do_unrolled_row4(float &cov1, float &cov2, float &cov3, float &cov4,
                                         float &distcol1, float &distcol2, float &distcol3,
                                         float &distcol4, unsigned int &idxcol1,
                                         unsigned int &idxcol2, unsigned int &idxcol3, unsigned int &idxcol4,
                                         const float &inormcx, const float &inormcy, const float &inormcz,
                                         const float &inormcw, const float &inormr,
                                         const float &df_colx, const float &df_coly, const float &df_colz,
                                         const float &df_colw, const float &dg_colx, const float &dg_coly,
                                         const float &dg_colz, const float &dg_colw, const float &df_row,
                                         const float &dg_row, const int &row, const int &col,
                                         const int &global_row, const int &global_col,
                                         mp_entry* __restrict__ mp_row, const float &curr_val) {

    float4 dist;

    // Compute the row's distances
    dist.x = static_cast<float>(cov1) * inormcx * inormr;
    dist.y = static_cast<float>(cov2) * inormcy * inormr;
    dist.z = static_cast<float>(cov3) * inormcz * inormr;
    dist.w = static_cast<float>(cov4) * inormcw * inormr;

    // Compute the next covariance values
    cov1 = cov1 + df_colx * dg_row + dg_colx * df_row;
    cov2 = cov2 + df_coly * dg_row + dg_coly * df_row;
    cov3 = cov3 + df_colz * dg_row + dg_colz * df_row;
    cov4 = cov4 + df_colw * dg_row + dg_colw * df_row;

    // Update the column best-so-far values
    MPMax2(distcol1, dist.x, idxcol1, global_row);
    MPMax2(distcol2, dist.y, idxcol2, global_row);
    MPMax2(distcol3, dist.z, idxcol3, global_row);
    MPMax2(distcol4, dist.w, idxcol4, global_row);
    unsigned int idx;

    // We take the maximum of the columns we computed for the row
    // And use that value to check the matrix profile
    float d = max4(dist, global_col, idx);
    MPatomicMax_check((unsigned long long*) (mp_row + row), d, idx, curr_val);
}

// Processes 4 iterations of the inner loop. Each thread computes 4 distances per iteration (x,y), (x+1,y), (x+2,y), and (x+3,y)
// This function assumes that the edge cases that occur on the edge of the distance matrix are not present. This is the faster path,
// with less conditional branching.
__device__ inline void do_iteration_unroll_4(int i, int j, int x, int y, float &cov1, float &cov2, float &cov3, float &cov4,
                                             float* __restrict__ df_col, float* __restrict__ df_row, float* __restrict__ dg_col,
                                             float* __restrict__ dg_row, float* __restrict__ inorm_col, float* __restrict__ inorm_row,
                                             mp_entry* __restrict__ local_mp_col, mp_entry* __restrict__ local_mp_row) 
{
    float4 distc = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);
    float4 distc2 = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);
    uint4 idxc,idxc2;
    
    // Load row values 2 at a time, load column values 4 at a time
    int r = i >> 1;
    int c = j >> 2;
    int c2 = j >> 1;

    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a float4 vector type
    float4 dfc = reinterpret_cast<float4*>(df_col)[c];
    float4 dgc = reinterpret_cast<float4*>(dg_col)[c];
    float4 inormc = (reinterpret_cast<float4*>(inorm_col)[c]);
    float4 dfc2 = reinterpret_cast<float4*>(df_col)[c+1];
    float4 dgc2 = reinterpret_cast<float4*>(dg_col)[c+1];
    float4 inormc2 = reinterpret_cast<float4*>(inorm_col)[c+1];

    // Copy the pieces of the cache we will use into registers with vectorized loads
    ulonglong2 mp_col_check1 = reinterpret_cast<ulonglong2*>(local_mp_col)[c2];
    ulonglong2 mp_col_check2 = reinterpret_cast<ulonglong2*>(local_mp_col)[c2 + 1];

    // Due to a lack of registers on volta, we only load these row values 2 at a time
    float2 dgr = reinterpret_cast<float2*>(dg_row)[r];
    float2 dfr = reinterpret_cast<float2*>(df_row)[r];
    float2 inormr = reinterpret_cast<float2*>(inorm_row)[r];
    ulonglong2 mp_row_check = reinterpret_cast<ulonglong2*>(local_mp_row)[r];

    // Do rows one at a time:
    // We are computing a tile that looks like this:
    // C:1 2 3 4 5 6 7
    //R1 X X X X
    //R2   X X X X
    //R3     X X X X
    //R4       X X X X
    // For 4 diagonals unrolled 4 times we compute a total of 16 distances.
    // These distances cover 4 possible rows and 7 possible columns, so we need to check the matrix profile
    // 11 times total, once for each row and once for each column
    mp_entry e;
    e.ulong = mp_row_check.x; 
    do_unrolled_row4(cov1, cov2, cov3, cov4, distc.x, distc.y, distc.z, distc.w,
                     idxc.x, idxc.y, idxc.z, idxc.w, inormc.x, inormc.y, inormc.z, 
                     inormc.w, inormr.x, dfc.x, dfc.y, dfc.z, dfc.w, dgc.x, dgc.y,
                     dgc.z, dgc.w, dfr.x, dgr.x, i, j, y, x, local_mp_row, e.floats[0]);
    e.ulong = mp_col_check1.x;

    // Each row's computation allows us to complete a column, the first row completes column 1
    MPatomicMax_check((unsigned long long*) (local_mp_col + j), distc.x, idxc.x, e.floats[0]);

    e.ulong = mp_row_check.y;
    do_unrolled_row4(cov1, cov2, cov3, cov4, distc.y, distc.z, distc.w, distc2.x,
                     idxc.y, idxc.z, idxc.w, idxc2.x, inormc.y, inormc.z, inormc.w,
                     inormc2.x, inormr.y, dfc.y, dfc.z, dfc.w, dfc2.x, dgc.y, dgc.z,
                     dgc.w, dgc2.x, dfr.y, dgr.y, i + 1, j + 1, y + 1, x + 1,
                     local_mp_row, e.floats[0]);

    // The second row completes column 2
    e.ulong = mp_col_check1.y;
    MPatomicMax_check((unsigned long long*) (local_mp_col + j + 1), distc.y, idxc.y, e.floats[0]);

    // Load the values for the next 2 rows
    dgr = reinterpret_cast<float2*>(dg_row)[r + 1];
    dfr = reinterpret_cast<float2*>(df_row)[r + 1];
    inormr = reinterpret_cast<float2*>(inorm_row)[r + 1];
    mp_row_check = reinterpret_cast<ulonglong2*>(local_mp_row)[r + 1];

    e.ulong  = mp_row_check.x;
    do_unrolled_row4(cov1, cov2, cov3, cov4, distc.z, distc.w, distc2.x, distc2.y,
                     idxc.z, idxc.w, idxc2.x, idxc2.y, inormc.z, inormc.w, inormc2.x,
                     inormc2.y, inormr.x, dfc.z, dfc.w, dfc2.x, dfc2.y, dgc.z, dgc.w,
                     dgc2.x, dgc2.y, dfr.x, dgr.x, i + 2, j + 2, y + 2, x + 2,
                     local_mp_row, e.floats[0]);

   
    // The third row completes column 3
    e.ulong  = mp_col_check2.x;
    MPatomicMax_check((unsigned long long*) (local_mp_col + j + 2), distc.z, idxc.z, e.floats[0]);
    
    e.ulong = mp_row_check.y;
    do_unrolled_row4(cov1, cov2, cov3, cov4, distc.w, distc2.x, distc2.y, distc2.z,
                     idxc.w, idxc2.x, idxc2.y, idxc2.z, inormc.w, inormc2.x, inormc2.y,
                     inormc2.z, inormr.y, dfc.w, dfc2.x, dfc2.y, dfc2.z, dgc.w, dgc2.x,
                     dgc2.y, dgc2.z, dfr.y, dgr.y, i + 3, j + 3, y + 3, x + 3,
                     local_mp_row, e.floats[0]);
    
    // After the 4th row, we have completed columns 4, 5, 6, and 7
    e.ulong = mp_col_check2.y;
    mp_col_check1 = reinterpret_cast<ulonglong2*>(local_mp_col)[c2+2];
    mp_col_check2 = reinterpret_cast<ulonglong2*>(local_mp_col)[c2+3];
    MPatomicMax_check((unsigned long long*) (local_mp_col + j + 3), distc.w, idxc.w, e.floats[0]);
    e.ulong = mp_col_check1.x;
    MPatomicMax_check((unsigned long long*) (local_mp_col + j + 4), distc2.x, idxc2.x, e.floats[0]);
    e.ulong = mp_col_check1.y;
    MPatomicMax_check((unsigned long long*) (local_mp_col + j + 5), distc2.y, idxc2.y, e.floats[0]);
    e.ulong = mp_col_check2.x;
    MPatomicMax_check((unsigned long long*) (local_mp_col + j + 6), distc2.z, idxc2.z, e.floats[0]);
    
}


// Does a single iteration of the inner loop on 4 diagonals per thread, not unrolled
// Checks for the boundary case where only 1, 2, or 3 diagonals can be updated
__device__ inline void do_iteration_4diag(int i, int j, int x, int y,
                                          size_t global_start_x, size_t global_start_y,
                                          int n, float &cov1, float &cov2,
                                          float &cov3, float &cov4, float *df_col, float *df_row,
                                          float *dg_col, float *dg_row, float *inorm_col, float *inorm_row,
                                          mp_entry *local_mp_col, mp_entry *local_mp_row, size_t diag, size_t num_diags)
{
    float dist_1;
    unsigned int idx_1;
    float4 dist;
    // Compute the next set of distances (row y)
    dist.x = static_cast<float>(cov1) * inorm_col[j] * inorm_row[i];
    dist.y = static_cast<float>(cov2) * inorm_col[j + 1] * inorm_row[i];
    dist.z = static_cast<float>(cov3) * inorm_col[j + 2] * inorm_row[i];
    dist.w = static_cast<float>(cov4) * inorm_col[j + 3] * inorm_row[i];

    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + df_col[j] * dg_row[i] + dg_col[j] * df_row[i];
    cov2 = cov2 + df_col[j+1] * dg_row[i] + dg_col[j+1] * df_row[i];
    cov3 = cov3 + df_col[j+2] * dg_row[i] + dg_col[j + 2] * df_row[i];
    cov4 = cov4 + df_col[j+3] * dg_row[i] + dg_col[j + 3] * df_row[i];

    MPatomicMax((unsigned long long*) (local_mp_col + j), dist.x, y);
    dist_1 = dist.x;
    idx_1 = x + global_start_x;
    if(x + 1 < n && diag + 1 < num_diags) {
        MPMax(dist_1, dist.y, idx_1, global_start_x + x + 1, dist_1, idx_1);
        MPatomicMax((unsigned long long*) (local_mp_col + j + 1), dist.y, y + global_start_y);
    }
    if(x + 2 < n && diag + 2 < num_diags) {
        MPMax(dist_1, dist.z, idx_1, global_start_x + x + 2, dist_1, idx_1);
        MPatomicMax((unsigned long long*) (local_mp_col + j + 2), dist.z, y + global_start_y);
    }
    if(x + 3 < n && diag + 3 < num_diags) {
        MPMax(dist_1, dist.w, idx_1, global_start_x + x + 3, dist_1, idx_1);
        MPatomicMax((unsigned long long*) (local_mp_col + j + 3), dist.w, y + global_start_y);
    }
    MPatomicMax((unsigned long long*) (local_mp_row + i), dist_1, idx_1);	
}

//Computes the matrix profile given the sliding dot products for the first query and the precomputed data statisics
__global__ void __launch_bounds__(BLOCKSZ, (BLOCKSZ * BLOCKSPERSM)  / BLOCKSZ)
do_tile_self_join(const double* __restrict__ Cov, const double* __restrict__ dfa,
                  const double* __restrict__ dfb, const double* __restrict__ dga,
                  const double* __restrict__ dgb, const double* __restrict__ normsa,
                  const double* __restrict__ normsb,
                  unsigned long long* __restrict__ profile_A,
                  unsigned long long* __restrict__ profile_B,
                  const unsigned int m, const unsigned int n_x, const unsigned int n_y,
                  const unsigned int global_start_x, const unsigned int global_start_y,
                  const int exclusion_lower, const int exclusion_upper)
{
    // tile_height must be a multiple of 4
    // Tuned for V100
    const int tile_height = TILE_HEIGHT;
    const int tile_width = tile_height + BLOCKSZ * 4;
    __shared__ mp_entry local_mp_col[tile_width];
    __shared__ mp_entry local_mp_row[tile_height];
    __shared__ float df_col[tile_width];
    __shared__ float dg_col[tile_width];
    __shared__ float inorm_col[tile_width];
    __shared__ float df_row[tile_height];
    __shared__ float dg_row[tile_height];
    __shared__ float inorm_row[tile_height];

    const unsigned int start_diag = (threadIdx.x << 2) + blockIdx.x * (blockDim.x << 2);

    // This is the index of the meta-diagonal that this thread block will work on
    const unsigned int meta_diagonal_idx = blockIdx.x;

    // The first threads are acutally computing the trivial match between the same subsequence
    // we exclude these from the calculation
    int tile_start_x = meta_diagonal_idx * (BLOCKSZ * 4) + exclusion_lower;
    int tile_start_y = 0;
      
    // x is the global column of the distance matrix
    // y is the global row of the distance matrix
    // localX, localY are the local coordinates of the thread position in the tile it is working on
    int x = tile_start_x + threadIdx.x * 4;
    int y = 0;

    // Each thread updates 2 diagonals at once
    float cov1, cov2, cov3, cov4;
    
    const unsigned int num_diags = n_x - exclusion_upper;
    
    // Load the first dot product values
    if (x < n_x) {
        cov1 = Cov[x];
    }
    
    if (x + 1 < n_x) {
        cov2 = Cov[x + 1];
    }
    
    if (x + 2 < n_x) {
        cov3 = Cov[x + 2];
    }

    if(x + 3 < n_x) {
        cov4 = Cov[x + 3]; 
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
    while (tile_start_x < n_x && tile_start_y < n_y)
    {
        // Initialize the next tile's shared memory
        initialize_tile_memory<tile_height,tile_width>(profile_A, profile_B, dfa, dfb, dga, dgb,
                                                       normsa, normsb, local_mp_col, local_mp_row,
                                                       df_col, df_row, dg_col, dg_row, inorm_col, inorm_row,
                                                       n_x, n_y, tile_start_x, tile_start_y);
        // Start of new tile, sync
        __syncthreads();

        // There are 2 pathways here, most of the time we take the fast path (top),
        // the last block will take the slower path as well as the fast path (bottom)
        if(tile_start_x + tile_width < n_x && tile_start_y + tile_height < n_y && start_diag + 3 < num_diags) {
            for(int i = 0, j = threadIdx.x << 2; i < tile_height; i+=4, j+=4) {
                do_iteration_unroll_4(i,j,x + global_start_x + i,y + global_start_y + i, cov1,cov2,cov3,cov4,df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row);
            }
            x += tile_height;
            y += tile_height;
        } else if (start_diag < num_diags){
            int localX = threadIdx.x << 2;
            int localY = 0;
            while(x < n_x && y < n_y && localY < tile_height) {
                do_iteration_4diag(localY,localX,x,y,global_start_x,global_start_y,n_x,cov1,cov2,cov3,cov4, df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row, start_diag, num_diags);
                ++x;
                ++y;
                ++localX;
                ++localY;
            } 
        }

        // After this sync, the caches will be updated with the best so far values for this tile
        __syncthreads();

        // If we updated any values in the cached MP, try to push them to the global "master" MP
        int global_position = tile_start_x + threadIdx.x;
        int local_position = threadIdx.x;
        while(local_position < tile_width && global_position < n_x) {
            mp_entry e = local_mp_col[local_position];
        	MPatomicMax(profile_A + global_position, e.floats[0], e.ints[1]);
            global_position += BLOCKSZ;
            local_position += BLOCKSZ;
        }

        global_position = tile_start_y + threadIdx.x;
        local_position = threadIdx.x;
        while(local_position < tile_height && global_position < n_y) {
            mp_entry e = local_mp_row[local_position];
            MPatomicMax(profile_B + global_position, e.floats[0], e.ints[1]);
            global_position += BLOCKSZ;
            local_position += BLOCKSZ;
        }
        
        // Update the tile position
        tile_start_x += tile_height;
        tile_start_y += tile_height;

        // Make sure our updates were committed before we pull in the next tile
        __threadfence_block();
    }
    

}


SCRIMPError_t kernel_self_join_upper(const double *QT, const double *timeseries_A, const double *timeseries_B, const double *df_A, const double *df_B, const double *dg_A, const double *dg_B, const double *norms_A, const double *norms_B, unsigned long long int *profile_A, unsigned long long int *profile_B, size_t window_size, size_t tile_width, size_t tile_height, size_t global_x, size_t global_y, cudaStream_t s)
{
        dim3 grid(1,1,1);
        dim3 block(BLOCKSZ, 1, 1);
        int exclusion = window_size / 4;
        if(global_y >= global_x && global_y <= global_x + exclusion) {
            int num_workers = ceil((tile_width - exclusion) / 4.0);
            grid.x = ceil(num_workers / (double) BLOCKSZ);
        } else {
            int num_workers = ceil(tile_width / 4.0);
            grid.x = ceil(num_workers / (double) BLOCKSZ);
            exclusion = 0;
        }
        if(exclusion < tile_width) {
            do_tile_self_join<<<grid,block, 0,s>>>(QT,df_A,df_B,dg_A,dg_B,norms_A,norms_B,profile_A, profile_B,
                                               window_size, tile_width, tile_height, global_x, global_y,
                                               exclusion,0);
        }
        cudaError_t err = cudaPeekAtLastError();
        if(err != cudaSuccess) {
            return SCRIMP_CUDA_ERROR;
        }
        return SCRIMP_NO_ERROR;
}

SCRIMPError_t kernel_self_join_lower(const double *QT, const double *timeseries_A, const double *timeseries_B, const double *df_A, const double *df_B, const double *dg_A, const double *dg_B, const double *norms_A, const double *norms_B, unsigned long long int *profile_A, unsigned long long int *profile_B, size_t window_size, size_t tile_width, size_t tile_height, size_t global_x, size_t global_y, cudaStream_t s)
{
        dim3 grid(1,1,1);
        dim3 block(BLOCKSZ, 1, 1);
        int exclusion = window_size / 4;
        if(global_y + tile_height >= global_x && global_y + tile_height <= global_x + exclusion) {
            int num_workers = ceil((tile_height - exclusion) / 4.0);
            grid.x = ceil(num_workers / (double) BLOCKSZ);
        } else {
            int num_workers = ceil(tile_height / 4.0);
            grid.x = ceil(num_workers / (double) BLOCKSZ);
            exclusion = 0;
        }
        if(exclusion < tile_height) {
            do_tile_self_join<<<grid,block, 0,s>>>(QT,df_B,df_A,dg_B,dg_A,norms_B,norms_A,profile_B, profile_A,
                                               window_size, tile_height, tile_width, global_y, global_x,
                                               0, exclusion);
        }
        cudaError_t err = cudaPeekAtLastError();
        if(err != cudaSuccess) {
            return SCRIMP_CUDA_ERROR;
        }
        return SCRIMP_NO_ERROR;



}

} // namespace SCRIMP
