#include "kernels.h"

namespace SCRIMP {

#define BLOCKSZ_SP 512
#define BLOCKSZ_DP 256
#define BLOCKSPERSM_SELF 2
#define BLOCKSPERSM_AB 2
#define TILE_HEIGHT 200
#define TILE_HEIGHT_DP 200

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

template<class T, int tile_height, int tile_width, bool full_join, bool only_col, int BLOCKSZ>
__device__  inline void initialize_tile_memory(const unsigned long long int* __restrict__ profile_A,
                                        const unsigned long long int* __restrict__ profile_B,
                                        const double* __restrict__ df_A, const double* __restrict__ df_B,
                                        const double* __restrict__ dg_A, const double* __restrict__ dg_B,
                                        const double* __restrict__ norms_A, const double* __restrict__ norms_B,
                                        mp_entry* __restrict__ local_mp_col, mp_entry* __restrict__ local_mp_row,
                                        T* __restrict__ df_col, T* __restrict__ df_row, T* __restrict__ dg_col,
                                        T* __restrict__ dg_row, T* __restrict__ norm_col, T* __restrict__ norm_row,
                                        const unsigned int n_x, const unsigned int n_y, const unsigned int col_start,
                                        const unsigned int row_start)
{
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while(local_position < tile_width && global_position < n_x) {
        dg_col[local_position] = dg_A[global_position];
        df_col[local_position] = df_A[global_position];
        norm_col[local_position] = norms_A[global_position];
        if(full_join || only_col) {
            local_mp_col[local_position].ulong = profile_A[global_position];
        }
        local_position += BLOCKSZ;
        global_position += BLOCKSZ;
    }

    global_position = row_start + threadIdx.x;
    local_position = threadIdx.x;
    while(local_position < tile_height && global_position < n_y) {
        dg_row[local_position] = dg_B[global_position];
        df_row[local_position] = df_B[global_position];
        norm_row[local_position] = norms_B[global_position];
        if(full_join || !only_col) {
            local_mp_row[local_position].ulong = profile_B[global_position];
        }
        local_position += BLOCKSZ;
        global_position += BLOCKSZ;
    
    }

}

// This does one row of work for 2 diagonals in a single thread
template<class T, bool full_join, bool only_col>
__device__ inline void do_unrolled_row2(T &cov1, T &cov2, float &distcol1, float &distcol2, unsigned int &idxcol1,
                                        unsigned int &idxcol2, const T &inormcx, const T &inormcy, const T &inormr,
                                        const T &df_colx, const T &df_coly, const T &dg_colx, const T &dg_coly,
                                        const T &df_row, const T &dg_row, const int &row, const int &col,
                                        const int &global_row, const int &global_col, mp_entry* __restrict__ mp_row,
                                        const float &curr_val) {

    float2 dist;

    // Compute the row's distances
    dist.x = cov1 * inormcx * inormr;
    dist.y = cov2 * inormcy * inormr;

    // Compute the next covariance values
    cov1 = cov1 + df_colx * dg_row + dg_colx * df_row;
    cov2 = cov2 + df_coly * dg_row + dg_coly * df_row;

    
    // Update the column best-so-far values
    if(full_join || only_col) {
        MPMax2(distcol1, dist.x, idxcol1, global_row);
        MPMax2(distcol2, dist.y, idxcol2, global_row);
    }

    if(full_join || !only_col) {
        unsigned int idx = global_col;
        // We take the maximum of the columns we computed for the row
        // And use that value to check the matrix profile
        MPMax2(dist.x, dist.y, idx, global_col + 1);
        MPatomicMax_check((unsigned long long*) (mp_row + row), dist.x, idx, curr_val);
    }
}

// This does one row of work for 4 diagonals in a single thread
template<class T, bool full_join, bool only_col>
__device__ inline void do_unrolled_row4(T &cov1, T &cov2, T &cov3, T &cov4,
                                        float &distcol1, float &distcol2, float &distcol3,
                                        float &distcol4, unsigned int &idxcol1,
                                        unsigned int &idxcol2, unsigned int &idxcol3, unsigned int &idxcol4,
                                        const T &inormcx, const T &inormcy, const T &inormcz,
                                        const T &inormcw, const T &inormr,
                                        const T &df_colx, const T &df_coly, const T &df_colz,
                                        const T &df_colw, const T &dg_colx, const T &dg_coly,
                                        const T &dg_colz, const T &dg_colw, const T &df_row,
                                        const T &dg_row, const int &row, const int &col,
                                        const int &global_row, const int &global_col,
                                        mp_entry* __restrict__ mp_row, const float &curr_val) {

    float4 dist;

    // Compute the row's distances
    dist.x = cov1 * inormcx * inormr;
    dist.y = cov2 * inormcy * inormr;
    dist.z = cov3 * inormcz * inormr;
    dist.w = cov4 * inormcw * inormr;

    // Compute the next covariance values
    cov1 = cov1 + df_colx * dg_row + dg_colx * df_row;
    cov2 = cov2 + df_coly * dg_row + dg_coly * df_row;
    cov3 = cov3 + df_colz * dg_row + dg_colz * df_row;
    cov4 = cov4 + df_colw * dg_row + dg_colw * df_row;

    
    // Update the column best-so-far values
    if(full_join || only_col) {
        MPMax2(distcol1, dist.x, idxcol1, global_row);
        MPMax2(distcol2, dist.y, idxcol2, global_row);
        MPMax2(distcol3, dist.z, idxcol3, global_row);
        MPMax2(distcol4, dist.w, idxcol4, global_row);
    }

    if(full_join || !only_col) {
        unsigned int idx;
        // We take the maximum of the columns we computed for the row
        // And use that value to check the matrix profile
        float d = max4(dist, global_col, idx);
        MPatomicMax_check((unsigned long long*) (mp_row + row), d, idx, curr_val);
    }
}

// Processes 2 iterations of the inner loop. Each thread computes 2 distances per iteration (x,y), (x+1,y)
// This function assumes that the edge cases that occur on the edge of the distance matrix are not present. This is the faster path,
// with less conditional branching.
template<class T, class VT2, bool full_join, bool only_col>
__device__ inline void do_iteration_unroll_2(int i, int j, int x, int y, T &cov1, T &cov2,
                                             T* __restrict__ df_col, T* __restrict__ df_row,
                                             T* __restrict__ dg_col, T* __restrict__ dg_row,
                                             T* __restrict__ inorm_col, T* __restrict__ inorm_row,
                                             mp_entry* __restrict__ local_mp_col, mp_entry* __restrict__ local_mp_row) 
{
    float2 distc = make_float2(CC_MIN, CC_MIN);
    float2 distc2 = make_float2(CC_MIN, CC_MIN);
    uint2 idxc,idxc2;
    
    // Load row values 2 at a time, load column values 4 at a time
    int r = i >> 1;
    int c = j >> 1;

    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a float4 vector type
    VT2 dfc = reinterpret_cast<VT2*>(df_col)[c];
    VT2 dgc = reinterpret_cast<VT2*>(dg_col)[c];
    VT2 inormc = reinterpret_cast<VT2*>(inorm_col)[c];
    VT2 dfc2 = reinterpret_cast<VT2*>(df_col)[c+1];
    VT2 dgc2 = reinterpret_cast<VT2*>(dg_col)[c+1];
    VT2 inormc2 = reinterpret_cast<VT2*>(inorm_col)[c+1];
    ulonglong2 mp_col_check1, mp_col_check2;
    ulonglong2 mp_row_check;

    // Copy the pieces of the cache we will use into registers with vectorized loads
    if(full_join || only_col) {
        mp_col_check1 = reinterpret_cast<ulonglong2*>(local_mp_col)[c];
    }
    if(full_join || !only_col) {
        mp_row_check = reinterpret_cast<ulonglong2*>(local_mp_row)[r];
    }

    VT2 dgr = reinterpret_cast<VT2*>(dg_row)[r];
    VT2 dfr = reinterpret_cast<VT2*>(df_row)[r];
    VT2 inormr = reinterpret_cast<VT2*>(inorm_row)[r];
    
    // Do rows one at a time:
    // We are computing a tile that looks like this:
    // C:1 2 3
    //R1 X X
    //R2   X X
    // For 2 diagonals unrolled 2 times we compute a total of 4 distances.
    // These distances cover 2 possible rows and 3 possible columns, so we need to check the matrix profile
    // 5 times total, once for each row and once for each column
    mp_entry e;
    e.ulong = mp_row_check.x; 
    do_unrolled_row2<T, full_join,only_col>(cov1, cov2, distc.x, distc.y, idxc.x, idxc.y,
                                            inormc.x, inormc.y, inormr.x, dfc.x, dfc.y,
                                            dgc.x, dgc.y, dfr.x, dgr.x, i, j, y, x, local_mp_row, e.floats[0]);

    // Each row's computation allows us to complete a column, the first row completes column 1
    if(full_join || only_col) {
        e.ulong = mp_col_check1.x;
        MPatomicMax_check((unsigned long long*) (local_mp_col + j), distc.x, idxc.x, e.floats[0]);
    }

    e.ulong = mp_row_check.y;
    do_unrolled_row2<T,full_join, only_col>(cov1, cov2, distc.y, distc2.x, idxc.y,
                                            idxc2.x, inormc.y, inormc2.x, inormr.y,
                                            dfc.y, dfc2.x, dgc.y, dgc2.x, dfr.y, dgr.y,
                                            i + 1, j + 1, y + 1, x + 1, local_mp_row, e.floats[0]);

    // The second row completes column 2 and 3
    if(full_join || only_col) {
        e.ulong = mp_col_check1.y;
        MPatomicMax_check((unsigned long long*) (local_mp_col + j + 1), distc.y, idxc.y, e.floats[0]);
        mp_col_check2 = reinterpret_cast<ulonglong2*>(local_mp_col)[c+1];
        e.ulong = mp_col_check2.x;
        MPatomicMax_check((unsigned long long*) (local_mp_col + j + 2), distc2.x, idxc2.x, e.floats[0]);
    }
}

// Processes 4 iterations of the inner loop. Each thread computes 4 distances per iteration (x,y), (x+1,y), (x+2,y), and (x+3,y)
// This function assumes that the edge cases that occur on the edge of the distance matrix are not present. This is the faster path,
// with less conditional branching.
template<class T, class VT4, class VT2, bool full_join, bool only_col>
__device__ inline void do_iteration_unroll_4(int i, int j, int x, int y, T &cov1, T &cov2, T &cov3,
                                             T &cov4, T* __restrict__ df_col, T* __restrict__ df_row,
                                             T* __restrict__ dg_col, T* __restrict__ dg_row,
                                             T* __restrict__ inorm_col, T* __restrict__ inorm_row,
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
    VT4 dfc = reinterpret_cast<VT4*>(df_col)[c];
    VT4 dgc = reinterpret_cast<VT4*>(dg_col)[c];
    VT4 inormc = (reinterpret_cast<VT4*>(inorm_col)[c]);
    VT4 dfc2 = reinterpret_cast<VT4*>(df_col)[c+1];
    VT4 dgc2 = reinterpret_cast<VT4*>(dg_col)[c+1];
    VT4 inormc2 = reinterpret_cast<VT4*>(inorm_col)[c+1];
    ulonglong2 mp_col_check1, mp_col_check2;
    ulonglong2 mp_row_check;

    // Copy the pieces of the cache we will use into registers with vectorized loads
    if(full_join || only_col) {
        mp_col_check1 = reinterpret_cast<ulonglong2*>(local_mp_col)[c2];
    }
    if(full_join || !only_col) {
        mp_row_check = reinterpret_cast<ulonglong2*>(local_mp_row)[r];
    }

    // Due to a lack of registers on volta, we only load these row values 2 at a time
    VT2 dgr = reinterpret_cast<VT2*>(dg_row)[r];
    VT2 dfr = reinterpret_cast<VT2*>(df_row)[r];
    VT2 inormr = reinterpret_cast<VT2*>(inorm_row)[r];
    
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
    do_unrolled_row4<T, full_join,only_col>(cov1, cov2, cov3, cov4, distc.x, distc.y, distc.z, distc.w,
                     idxc.x, idxc.y, idxc.z, idxc.w, inormc.x, inormc.y, inormc.z, 
                     inormc.w, inormr.x, dfc.x, dfc.y, dfc.z, dfc.w, dgc.x, dgc.y,
                     dgc.z, dgc.w, dfr.x, dgr.x, i, j, y, x, local_mp_row, e.floats[0]);

    // Each row's computation allows us to complete a column, the first row completes column 1
    if(full_join || only_col) {
        e.ulong = mp_col_check1.x;
        MPatomicMax_check((unsigned long long*) (local_mp_col + j), distc.x, idxc.x, e.floats[0]);
    }

    e.ulong = mp_row_check.y;
    do_unrolled_row4<T,full_join, only_col>(cov1, cov2, cov3, cov4, distc.y, distc.z, distc.w, distc2.x,
                     idxc.y, idxc.z, idxc.w, idxc2.x, inormc.y, inormc.z, inormc.w,
                     inormc2.x, inormr.y, dfc.y, dfc.z, dfc.w, dfc2.x, dgc.y, dgc.z,
                     dgc.w, dgc2.x, dfr.y, dgr.y, i + 1, j + 1, y + 1, x + 1,
                     local_mp_row, e.floats[0]);

    // The second row completes column 2
    if(full_join || only_col) {
        e.ulong = mp_col_check1.y;
        MPatomicMax_check((unsigned long long*) (local_mp_col + j + 1), distc.y, idxc.y, e.floats[0]);
    }

    // Load the values for the next 2 rows
    dgr = reinterpret_cast<VT2*>(dg_row)[r + 1];
    dfr = reinterpret_cast<VT2*>(df_row)[r + 1];
    inormr = reinterpret_cast<VT2*>(inorm_row)[r + 1];

    if(full_join || !only_col) {
        mp_row_check = reinterpret_cast<ulonglong2*>(local_mp_row)[r + 1];
    }

    e.ulong  = mp_row_check.x;
    do_unrolled_row4<T,full_join,only_col>(cov1, cov2, cov3, cov4, distc.z, distc.w, distc2.x, distc2.y,
                     idxc.z, idxc.w, idxc2.x, idxc2.y, inormc.z, inormc.w, inormc2.x,
                     inormc2.y, inormr.x, dfc.z, dfc.w, dfc2.x, dfc2.y, dgc.z, dgc.w,
                     dgc2.x, dgc2.y, dfr.x, dgr.x, i + 2, j + 2, y + 2, x + 2,
                     local_mp_row, e.floats[0]);

   
    // The third row completes column 3
    if(full_join || only_col) {
        mp_col_check2 = reinterpret_cast<ulonglong2*>(local_mp_col)[c2 + 1];
        e.ulong  = mp_col_check2.x;
        MPatomicMax_check((unsigned long long*) (local_mp_col + j + 2), distc.z, idxc.z, e.floats[0]);
    }

    e.ulong = mp_row_check.y;
    do_unrolled_row4<T,full_join,only_col>(cov1, cov2, cov3, cov4, distc.w, distc2.x, distc2.y, distc2.z,
                     idxc.w, idxc2.x, idxc2.y, idxc2.z, inormc.w, inormc2.x, inormc2.y,
                     inormc2.z, inormr.y, dfc.w, dfc2.x, dfc2.y, dfc2.z, dgc.w, dgc2.x,
                     dgc2.y, dgc2.z, dfr.y, dgr.y, i + 3, j + 3, y + 3, x + 3,
                     local_mp_row, e.floats[0]);
    
    // After the 4th row, we have completed columns 4, 5, 6, and 7
    if(full_join || only_col) {
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
}

// Does a single iteration of the inner loop on 2 diagonals per thread, not unrolled
// Checks for the boundary case where only 1 diagonal can be updated
template<class T, bool full_join, bool only_col>
__device__ inline void do_iteration_2diag(int i, int j, int x, int y,
                                          size_t global_start_x, size_t global_start_y,
                                          int n, T &cov1, T &cov2, T* __restrict__ df_col,
                                          T* __restrict__ df_row, T* __restrict__ dg_col,
                                          T* __restrict__ dg_row, T* __restrict__ inorm_col,
                                          T* __restrict__ inorm_row, mp_entry* __restrict__ local_mp_col,
                                          mp_entry* __restrict__ local_mp_row, size_t diag, size_t num_diags)
{
    float dist_1;
    unsigned int idx_1;
    float2 dist;
    // Compute the next set of distances (row y)
    dist.x = cov1 * inorm_col[j] * inorm_row[i];
    dist.y = cov2 * inorm_col[j + 1] * inorm_row[i];

    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + df_col[j] * dg_row[i] + dg_col[j] * df_row[i];
    cov2 = cov2 + df_col[j+1] * dg_row[i] + dg_col[j+1] * df_row[i];

    if(full_join || only_col) {
        MPatomicMax((unsigned long long*) (local_mp_col + j), dist.x, y + global_start_y);
    }
    dist_1 = dist.x;
    idx_1 = x + global_start_x;
    if(x + 1 < n && diag + 1 < num_diags) {
        if(full_join || !only_col) {
            MPMax(dist_1, dist.y, idx_1, global_start_x + x + 1, dist_1, idx_1);
        }
        if(full_join || only_col) {
            MPatomicMax((unsigned long long*) (local_mp_col + j + 1), dist.y, y + global_start_y);
        }
    }
    if(full_join || !only_col) {
        MPatomicMax((unsigned long long*) (local_mp_row + i), dist_1, idx_1);	
    }
}

// Does a single iteration of the inner loop on 4 diagonals per thread, not unrolled
// Checks for the boundary case where only 1, 2, or 3 diagonals can be updated
template<class T, bool full_join, bool only_col>
__device__ inline void do_iteration_4diag(int i, int j, int x, int y,
                                          size_t global_start_x, size_t global_start_y,
                                          int n, T &cov1, T &cov2,
                                          T &cov3, T &cov4, T* __restrict__ df_col,
                                          T* __restrict__ df_row, T* __restrict__ dg_col,
                                          T* __restrict__ dg_row, T* __restrict__ inorm_col,
                                          T* __restrict__ inorm_row, mp_entry* __restrict__ local_mp_col,
                                          mp_entry* __restrict__ local_mp_row, size_t diag, size_t num_diags)
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

    if(full_join || only_col) {
        MPatomicMax((unsigned long long*) (local_mp_col + j), dist.x, y + global_start_y);
    }
    dist_1 = dist.x;
    idx_1 = x + global_start_x;
    if(x + 1 < n && diag + 1 < num_diags) {
        if(full_join || !only_col) {
            MPMax(dist_1, dist.y, idx_1, global_start_x + x + 1, dist_1, idx_1);
        }
        if(full_join || only_col) {
            MPatomicMax((unsigned long long*) (local_mp_col + j + 1), dist.y, y + global_start_y);
        }
    }
    if(x + 2 < n && diag + 2 < num_diags) {
        if(full_join || !only_col) {
            MPMax(dist_1, dist.z, idx_1, global_start_x + x + 2, dist_1, idx_1);
        }
        if(full_join || only_col) {
            MPatomicMax((unsigned long long*) (local_mp_col + j + 2), dist.z, y + global_start_y);
        }
    }
    if(x + 3 < n && diag + 3 < num_diags) {
        if(full_join || !only_col) {
            MPMax(dist_1, dist.w, idx_1, global_start_x + x + 3, dist_1, idx_1);
        }
        if(full_join || only_col) {
            MPatomicMax((unsigned long long*) (local_mp_col + j + 3), dist.w, y + global_start_y);
        }
    }
    if(full_join || !only_col) {
        MPatomicMax((unsigned long long*) (local_mp_row + i), dist_1, idx_1);	
    }
}

//Computes the matrix profile given the sliding dot products for the first query and the precomputed data statisics
template<class T, class T2, class T4, bool fp64, bool full_join, bool only_col, int blocks_per_sm, int diags_per_thread, int tile_height, int BLOCKSZ>
__global__ void __launch_bounds__(BLOCKSZ, blocks_per_sm)
do_tile(const double* __restrict__ Cov, const double* __restrict__ dfa,
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
    const int tile_width = tile_height + BLOCKSZ * diags_per_thread;
    extern __shared__ char smem[];
    mp_entry *local_mp_col, *local_mp_row;
    T *df_col, *dg_col, *inorm_col, *df_row, *dg_row, *inorm_row;
    
    
    df_col = (T*) smem;
    dg_col = df_col + tile_width;
    inorm_col = dg_col + tile_width;
    df_row = inorm_col + tile_width;
    dg_row = df_row + tile_height;
    inorm_row = dg_row + tile_height;
    mp_entry *pos = (mp_entry*) (inorm_row + tile_height);
    
    if(!full_join && only_col) {
        local_mp_col = pos;
    } else if(!full_join) {
        local_mp_row = pos;
    } else {
        local_mp_col = pos;
        local_mp_row = pos + tile_width;
    }

    const unsigned int start_diag = (threadIdx.x * diags_per_thread) + blockIdx.x * (blockDim.x * diags_per_thread);

    // This is the index of the meta-diagonal that this thread block will work on
    const unsigned int meta_diagonal_idx = blockIdx.x;

    // The first threads are acutally computing the trivial match between the same subsequence
    // we exclude these from the calculation
    int tile_start_x = meta_diagonal_idx * (BLOCKSZ * diags_per_thread) + exclusion_lower;
    int tile_start_y = 0;
      
    // x is the global column of the distance matrix
    // y is the global row of the distance matrix
    // localX, localY are the local coordinates of the thread position in the tile it is working on
    int x = tile_start_x + threadIdx.x * diags_per_thread;
    int y = 0;

    // Each thread updates 2 diagonals at once
    T cov1, cov2, cov3, cov4;
    
    const unsigned int num_diags = n_x - exclusion_upper;
    
    // Load the first dot product values
    if (x < n_x) {
        cov1 = Cov[x];
    }
    
    if (x + 1 < n_x && diags_per_thread > 1) {
        cov2 = Cov[x + 1];
    }
    
    if (x + 2 < n_x && diags_per_thread > 2) {
        cov3 = Cov[x + 2];
    }

    if(x + 3 < n_x && diags_per_thread > 3) {
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
        initialize_tile_memory<T,tile_height,tile_width,full_join,only_col, BLOCKSZ>(profile_A, profile_B, dfa, dfb, dga, dgb,
                                                       normsa, normsb, local_mp_col, local_mp_row,
                                                       df_col, df_row, dg_col, dg_row, inorm_col, inorm_row,
                                                       n_x, n_y, tile_start_x, tile_start_y);
        // Start of new tile, sync
        __syncthreads();

        // There are 2 pathways here, most of the time we take the fast path (top),
        // the last block will take the slower path as well as the fast path (bottom)
        if(tile_start_x + tile_width < n_x && tile_start_y + tile_height < n_y && start_diag + diags_per_thread - 1 < num_diags) {
            for(int i = 0, j = threadIdx.x * diags_per_thread; i < tile_height; i+=diags_per_thread, j+=diags_per_thread) {
                if(diags_per_thread == 4) {
                    do_iteration_unroll_4<T,T4,T2,full_join, only_col>(i,j,x + global_start_x + i,y + global_start_y + i, cov1,cov2,cov3,cov4,df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row);
                } else if(diags_per_thread == 2) {
                    do_iteration_unroll_2<T,T2,full_join, only_col>(i,j,x + global_start_x + i,y + global_start_y + i, cov1,cov2,df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row);
                }
            }                
            x += tile_height;
            y += tile_height;
        } else if (start_diag < num_diags){
            int localX = threadIdx.x * diags_per_thread;
            int localY = 0;
            while(x < n_x && y < n_y && localY < tile_height) {
                if(diags_per_thread == 4) {
                    do_iteration_4diag<T,full_join, only_col>(localY,localX,x,y,global_start_x,global_start_y,n_x,cov1,cov2,cov3,cov4, df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row, start_diag, num_diags);
                } else if(diags_per_thread == 2) {
                    do_iteration_2diag<T,full_join, only_col>(localY,localX,x,y,global_start_x,global_start_y,n_x,cov1,cov2,df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row, start_diag, num_diags);
                }
                ++x;
                ++y;
                ++localX;
                ++localY;
            } 
        }

        // After this sync, the caches will be updated with the best so far values for this tile
        __syncthreads();

        int global_position, local_position;
        if(full_join || only_col) {
            // If we updated any values in the cached MP, try to push them to the global "master" MP
            global_position = tile_start_x + threadIdx.x;
            local_position = threadIdx.x;
            while(local_position < tile_width && global_position < n_x) {
                mp_entry e = local_mp_col[local_position];
                MPatomicMax(profile_A + global_position, e.floats[0], e.ints[1]);
                global_position += BLOCKSZ;
                local_position += BLOCKSZ;
            }
        }
        if(full_join || !only_col) {
            global_position = tile_start_y + threadIdx.x;
            local_position = threadIdx.x;
            while(local_position < tile_height && global_position < n_y) {
                mp_entry e = local_mp_row[local_position];
                MPatomicMax(profile_B + global_position, e.floats[0], e.ints[1]);
                global_position += BLOCKSZ;
                local_position += BLOCKSZ;
            }
        }
        
        // Update the tile position
        tile_start_x += tile_height;
        tile_start_y += tile_height;

        // Make sure our updates were committed before we pull in the next tile
        __threadfence_block();
    }
    

}

int get_diags_per_thread(bool fp64, const cudaDeviceProp &dev_prop) {
    return 4;
}

int get_blocksz(bool fp64, const cudaDeviceProp &dev_prop) {
    if(fp64) {
        return BLOCKSZ_DP;
    } else {
        return BLOCKSZ_SP;
    }
}

template< class T >
int get_smem(int tile_height, bool fp64, bool full_join, bool only_column_join, const cudaDeviceProp &dev_prop) {
    int smem;
    int diags_per_thread = get_diags_per_thread(fp64, dev_prop);
    int blocksz = get_blocksz(fp64, dev_prop);
    int tile_width = blocksz * diags_per_thread + tile_height;
    smem = (tile_width + tile_height) * 3 * sizeof(T);
    if(full_join) {
        smem += (tile_width + tile_height) * sizeof(mp_entry);
    } else if( only_column_join){
        smem += tile_width * sizeof(mp_entry);
    } else {
        smem += tile_height * sizeof(mp_entry);
    }
    printf("Using %d KiB smem per block\n", smem / 1024);
    return smem;
} 


SCRIMPError_t kernel_ab_join_upper(const double *QT, const double *timeseries_A, const double *timeseries_B, const double *df_A, const double *df_B, const double *dg_A, const double *dg_B, const double *norms_A, const double *norms_B, unsigned long long int *profile_A, unsigned long long int *profile_B, size_t window_size, size_t tile_width, size_t tile_height, size_t global_x, size_t global_y, size_t global_start_x, size_t global_start_y, const cudaDeviceProp &props, bool fp64, bool full_join, cudaStream_t s)
{
        int diags_per_thread = get_diags_per_thread(fp64, props);
        int blocksz = get_blocksz(fp64, props);
        dim3 grid(1,1,1);
        dim3 block(blocksz, 1, 1);
        int num_workers = ceil(tile_width / (float) diags_per_thread);
        grid.x = ceil(num_workers / (double) blocksz);
        if(full_join) {
            // We can have an exclusion zone if this ab join is part of a larger self-join
            int exclusion = window_size / 4;
            if(global_y + global_start_y >= global_x + global_start_x && global_start_y + global_y <= global_start_x + global_x + exclusion) {
                num_workers = ceil((tile_width - exclusion) / (float) diags_per_thread);
                grid.x = ceil(num_workers / (double) blocksz);
            }else {
                exclusion = 0;
            }
            if(tile_width <= exclusion) {
                return SCRIMP_NO_ERROR
            }
            if(fp64) {
                int smem = get_smem<double>(TILE_HEIGHT_DP, fp64, true, true, props);
                do_tile<double, double2, double4, true, true, true, BLOCKSPERSM_AB, 4, TILE_HEIGHT_DP, BLOCKSZ_DP><<<grid,block,smem,s>>>(QT,df_A,df_B,dg_A,dg_B,norms_A,norms_B,profile_A, profile_B,
                                                   window_size, tile_width, tile_height, global_x, global_y,
                                                   exclusion,0);
            
            } else {
                int smem = get_smem<float>(TILE_HEIGHT, fp64, true, true, props);
                do_tile<float, float2, float4, false, true, true, BLOCKSPERSM_AB, 4, TILE_HEIGHT, BLOCKSZ_SP><<<grid,block,smem,s>>>(QT,df_A,df_B,dg_A,dg_B,norms_A,norms_B,profile_A, profile_B,
                                                   window_size, tile_width, tile_height, global_x, global_y,
                                                   exclusion,0);
            }
        } else {
            if(fp64) {
                int smem = get_smem<double>(TILE_HEIGHT_DP, fp64, false, true, props);
                do_tile<double, double2, double4, true, false, true, BLOCKSPERSM_AB, 4, TILE_HEIGHT_DP, BLOCKSZ_DP><<<grid,block,smem,s>>>(QT,df_A,df_B,dg_A,dg_B,norms_A,norms_B,profile_A, profile_B,
                                                   window_size, tile_width, tile_height, global_x, global_y,
                                                   0,0);
            } else {
                int smem = get_smem<float>(TILE_HEIGHT, fp64, false, true, props);
                do_tile<float, float2, float4, false, false, true, BLOCKSPERSM_AB, 4, TILE_HEIGHT, BLOCKSZ_SP><<<grid,block,smem,s>>>(QT,df_A,df_B,dg_A,dg_B,norms_A,norms_B,profile_A, profile_B,
                                                   window_size, tile_width, tile_height, global_x, global_y,
                                                   0,0);
            }
        }
        cudaError_t err = cudaPeekAtLastError();
        if(err != cudaSuccess) {
            return SCRIMP_CUDA_ERROR;
        }
        return SCRIMP_NO_ERROR;
}

SCRIMPError_t kernel_ab_join_lower(const double *QT, const double *timeseries_A, const double *timeseries_B, const double *df_A, const double *df_B, const double *dg_A, const double *dg_B, const double *norms_A, const double *norms_B, unsigned long long int *profile_A, unsigned long long int *profile_B, size_t window_size, size_t tile_width, size_t tile_height, size_t global_x, size_t global_y, size_t global_start_x, size_t global_start_y, const cudaDeviceProp &props, bool fp64, bool full_join, cudaStream_t s)
{
        int diags_per_thread = get_diags_per_thread(fp64, props);
        int blocksz = get_blocksz(fp64, props);
        dim3 grid(1,1,1);
        dim3 block(blocksz, 1, 1);
        int num_workers = ceil(tile_height / (float) diags_per_thread);
        grid.x = ceil(num_workers / (double) blocksz);
        if(full_join) {
            // We can have an exclusion zone if this ab join is part of a larger self-join
            int exclusion = window_size / 4;
            if(global_y + global_start_y + tile_height >= global_x + global_start_x && global_y + global_start_y + tile_height <= global_x + global_start_x + exclusion) {
                num_workers = ceil((tile_height - exclusion) / (float) diags_per_thread);
                grid.x = ceil(num_workers / (double) blocksz);
            } else {
                exclusion = 0;
            }
            if(tile_height <= exclusion) {
                return SCRIMP_NO_ERROR;
            }
            if(fp64) {
                int smem = get_smem<double>(TILE_HEIGHT_DP, fp64, true, true, props);
                do_tile<double, double2, double4, true, true, true, BLOCKSPERSM_AB, 4, TILE_HEIGHT_DP, BLOCKSZ_DP><<<grid,block,smem,s>>>(QT,df_B,df_A,dg_B,dg_A,norms_B,norms_A,profile_B, profile_A,
                                                   window_size, tile_height, tile_width, global_y, global_x,
                                                   0,exclusion);

            } else {
                int smem = get_smem<float>(TILE_HEIGHT, fp64, true, true, props);
                do_tile<float, float2, float4, false, true, true, BLOCKSPERSM_AB, 4, TILE_HEIGHT, BLOCKSZ_SP><<<grid,block,smem,s>>>(QT,df_B,df_A,dg_B,dg_A,norms_B,norms_A,profile_B, profile_A,
                                                   window_size, tile_height, tile_width, global_y, global_x,
                                                   0,exclusion);
            }
        } else {
            if(fp64) {
                int smem = get_smem<double>(TILE_HEIGHT_DP, fp64, false, false, props);
                do_tile<double, double2, double4, true, false, false, BLOCKSPERSM_AB, 4, TILE_HEIGHT_DP, BLOCKSZ_DP><<<grid,block,smem,s>>>(QT,df_B,df_A,dg_B,dg_A,norms_B,norms_A,profile_B, profile_A,
                                                   window_size, tile_height, tile_width, global_y, global_x,
                                                   0,0);

            } else {
                int smem = get_smem<float>(TILE_HEIGHT, fp64, false, false, props);
                do_tile<float, float2, float4, false, false, false, BLOCKSPERSM_AB, 4, TILE_HEIGHT, BLOCKSZ_SP><<<grid,block,smem,s>>>(QT,df_B,df_A,dg_B,dg_A,norms_B,norms_A,profile_B, profile_A,
                                                   window_size, tile_height, tile_width, global_y, global_x,
                                                   0,0);
            }
        }
        cudaError_t err = cudaPeekAtLastError();
        if(err != cudaSuccess) {
            return SCRIMP_CUDA_ERROR;
        }
        return SCRIMP_NO_ERROR;



}

SCRIMPError_t kernel_self_join_upper(const double *QT, const double *timeseries_A, const double *timeseries_B, const double *df_A, const double *df_B, const double *dg_A, const double *dg_B, const double *norms_A, const double *norms_B, unsigned long long int *profile_A, unsigned long long int *profile_B, size_t window_size, size_t tile_width, size_t tile_height, size_t global_x, size_t global_y, const cudaDeviceProp &props, bool fp64, cudaStream_t s)
{
        int exclusion = window_size / 4;
        int diags_per_thread = get_diags_per_thread(fp64,props);
        int blocksz = get_blocksz(fp64,props);
        dim3 grid(1,1,1);
        dim3 block(blocksz, 1, 1);
        if(global_y >= global_x && global_y <= global_x + exclusion) {
            int num_workers = ceil((tile_width - exclusion) / (float) diags_per_thread);
            grid.x = ceil(num_workers / (double) blocksz);
        } else {
            int num_workers = ceil(tile_width / (float) diags_per_thread);
            grid.x = ceil(num_workers / (double) blocksz);
            exclusion = 0;
        }
        if(exclusion < tile_width) {
            if(fp64) {
                int smem = get_smem<double>(TILE_HEIGHT_DP, fp64, true, false, props);
                do_tile<double, double2, double4, true, true,false, BLOCKSPERSM_SELF, 4, TILE_HEIGHT_DP, BLOCKSZ_DP><<<grid,block,smem,s>>>(QT,df_A,df_B,dg_A,dg_B,norms_A,norms_B,profile_A, profile_B,
                                                   window_size, tile_width, tile_height, global_x, global_y,
                                                   exclusion,0);
            
            } else {
                int smem = get_smem<float>(TILE_HEIGHT, fp64, true, false, props);
                do_tile<float, float2, float4, false, true,false, BLOCKSPERSM_SELF,4, TILE_HEIGHT, BLOCKSZ_SP><<<grid,block,smem,s>>>(QT,df_A,df_B,dg_A,dg_B,norms_A,norms_B,profile_A, profile_B,
                                                   window_size, tile_width, tile_height, global_x, global_y,
                                                   exclusion,0);

            }
        }
        cudaError_t err = cudaPeekAtLastError();
        if(err != cudaSuccess) {
            return SCRIMP_CUDA_ERROR;
        }
        return SCRIMP_NO_ERROR;
}

SCRIMPError_t kernel_self_join_lower(const double *QT, const double *timeseries_A, const double *timeseries_B, const double *df_A, const double *df_B, const double *dg_A, const double *dg_B, const double *norms_A, const double *norms_B, unsigned long long int *profile_A, unsigned long long int *profile_B, size_t window_size, size_t tile_width, size_t tile_height, size_t global_x, size_t global_y, const cudaDeviceProp &props, bool fp64, cudaStream_t s)
{
        int exclusion = window_size / 4;
        int diags_per_thread = get_diags_per_thread(fp64, props);
        int blocksz = get_blocksz(fp64, props);
        dim3 grid(1,1,1);
        dim3 block(blocksz, 1, 1);
        if(global_y + tile_height >= global_x && global_y + tile_height <= global_x + exclusion) {
            int num_workers = ceil((tile_height - exclusion) / (float) diags_per_thread);
            grid.x = ceil(num_workers / (double) blocksz);
        } else {
            int num_workers = ceil(tile_height / (float) diags_per_thread);
            grid.x = ceil(num_workers / (double) blocksz);
            exclusion = 0;
        }
        if(exclusion < tile_height) {
            if(fp64) {
                int smem = get_smem<double>(TILE_HEIGHT_DP, fp64, true, false, props);
                do_tile<double, double2,double4, true, true,false, BLOCKSPERSM_SELF, 4, TILE_HEIGHT_DP, BLOCKSZ_DP><<<grid,block,smem,s>>>(QT,df_B,df_A,dg_B,dg_A,norms_B,norms_A,profile_B, profile_A,
                                               window_size, tile_height, tile_width, global_y, global_x,
                                               0, exclusion);
            } else {
                int smem = get_smem<float>(TILE_HEIGHT, fp64, true, false, props);
                do_tile<float,float2,float4, false, true,false, BLOCKSPERSM_SELF, 4, TILE_HEIGHT, BLOCKSZ_SP><<<grid,block,smem,s>>>(QT,df_B,df_A,dg_B,dg_A,norms_B,norms_A,profile_B, profile_A,
                                               window_size, tile_height, tile_width, global_y, global_x,
                                               0, exclusion);
            }
        }
        cudaError_t err = cudaPeekAtLastError();
        if(err != cudaSuccess) {
            return SCRIMP_CUDA_ERROR;
        }
        return SCRIMP_NO_ERROR;



}

} // namespace SCRIMP
