#include <vector>
#include <numeric>
#include <unordered_map>
#include <float.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/transform_scan.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <unistd.h>
#include "SCAMP.h"
#include "common.h"
#include "tile.h"

using std::vector;
using std::unordered_map;
using std::make_pair;

namespace SCAMP {

static const int ISSUED_ALL_DEVICES = -2;



__global__ void cross_correlation_to_ed(float *profile, unsigned int n, unsigned int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        profile[tid] = sqrt(max(2*(1 - profile[tid]), 0.0)) * sqrt((double)m);
    }
}

//This kernel computes a sliding mean with specified window size and a corresponding prefix sum array (A)
template<class DTYPE>
__global__ void sliding_mean(DTYPE* pref_sum,  size_t window, size_t size, DTYPE* means)
{
    const DTYPE coeff = 1.0 / (DTYPE) window;
    size_t a = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.x * blockDim.x + threadIdx.x + window;

    if(a == 0){
        means[a] = pref_sum[window - 1] * coeff;
    }
    if(a < size - 1){
        means[a + 1] = (pref_sum[b] - pref_sum[a]) * coeff;
    }
}

template<class DTYPE>
__global__ void sliding_norm(DTYPE* cumsumsqr, unsigned int window, unsigned int size, DTYPE* norms) {
    const DTYPE coeff = 1 / (DTYPE) window;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x * blockDim.x + threadIdx.x + window;
    if (a == 0) {
        norms[a] = 1 / sqrt(cumsumsqr[window - 1]);
    }
    else if (b < size + window) {
        norms[a] = 1 / sqrt(cumsumsqr[b - 1] - cumsumsqr[a - 1]);
    }
}

template<class DTYPE>
__global__ void sliding_dfdg(const DTYPE *T, const DTYPE *means, DTYPE *df, DTYPE *dg, const int m, const int n) {
    const DTYPE half = 1.0 / (DTYPE) 2.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n - 1) {
        df[tid] = (T[tid + m] - T[tid]) * half;
        dg[tid] = (T[tid + m] - means[tid + 1]) + (T[tid] - means[tid]);
    }
}

__global__ void __launch_bounds__(512,4) 
fastinvnorm(double *norm, const double *mean, const double *T, int m, int n) {
   
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int jump = ceil(n / (double) (blockDim.x * gridDim.x));
    int start = jump * tid;
    int end = jump * (tid + 1);
    end = min(end,n);
    if(start >= n) {
        return;
    }
    double sum = 0;
    for(int i = 0; i < m; ++i){ 
        double val = T[i + start] - mean[start];
        sum += val * val;
    }
    norm[start] = sum;
    
    for(int i = start+1; i < end; ++i) {
            norm[i] = norm[i - 1]  + ((T[i-1] - mean[i-1]) + (T[i + m - 1] - mean[i])) * (T[i + m - 1] - T[i - 1]);
    }
    for(int i = start; i < end; ++i) {
        norm[i] = 1.0 / sqrt(norm[i]);
    }
}


void compute_statistics(const double *T, double *norms, double *df, double *dg,
                        double *means, size_t n, size_t m, cudaStream_t s, double *scratch)
{
    square<double> sqr;
    dim3 grid(ceil(n / (double) 512), 1,1);
    dim3 block(512, 1, 1);
    
    gpuErrchk(cudaPeekAtLastError());
    
    thrust::device_ptr<const double> dev_ptr_T = thrust::device_pointer_cast(T);
    thrust::device_ptr<double> dev_ptr_scratch = thrust::device_pointer_cast(scratch);

    // Compute prefix sum in scratch
    thrust::inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1,
                           dev_ptr_scratch, thrust::plus<double>());
    gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum to compute sliding mean
    sliding_mean<double><<<grid, block, 0, s>>>(scratch, m, n, means);
    gpuErrchk(cudaPeekAtLastError());
    
    // Compute differential values
    sliding_dfdg<double><<<grid, block, 0, s>>>(T, means, df,dg,m,n);
    gpuErrchk(cudaPeekAtLastError());
    
    // Compute prefix sum of squares in scratch
    thrust::transform_inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1,
                                     dev_ptr_scratch, sqr,thrust::plus<double>());
    gpuErrchk(cudaPeekAtLastError());
    
    // This will be kind of slow on the GPU, may cause latency between tiles
    int workers = n / m + 1;
    fastinvnorm<<<dim3(ceil(workers / (double)512),1,1), dim3(512,1,1), 0, s>>>(norms, means, T, m, n);
    gpuErrchk(cudaPeekAtLastError());
    
}

SCAMPError_t SCAMP_Operation::init()
{
    for (auto device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        
        T_A_dev.insert(make_pair(device, (double*) 0));
        T_B_dev.insert(make_pair(device, (double*) 0));
        QT_dev.insert(make_pair(device, (double*) 0));
        means_A.insert(make_pair(device, (double*) 0));
        means_B.insert(make_pair(device, (double*) 0));
        norms_A.insert(make_pair(device, (double*) 0));
        norms_B.insert(make_pair(device, (double*) 0));
        df_A.insert(make_pair(device, (double*) 0));
        df_B.insert(make_pair(device, (double*) 0));
        dg_A.insert(make_pair(device, (double*) 0));
        dg_B.insert(make_pair(device, (double*) 0));
        profile_A_dev.insert(make_pair(device,(float*) NULL));
        profile_B_dev.insert(make_pair(device,(float*) NULL));
        profile_A_merged.insert(make_pair(device,(unsigned long long int*) NULL));
        profile_B_merged.insert(make_pair(device,(unsigned long long int*) NULL));
        profile_idx_A_dev.insert(make_pair(device,(unsigned int *) NULL));
        profile_idx_B_dev.insert(make_pair(device,(unsigned int *) NULL));
        scratchpad.insert(make_pair(device, (double*) NULL));

        cudaMalloc(&T_A_dev.at(device), sizeof(double) * tile_size);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&T_B_dev.at(device), sizeof(double) * tile_size);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_A_dev.at(device), sizeof(float) * tile_n_x);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_B_dev.at(device), sizeof(float) * tile_n_y);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_idx_A_dev.at(device), sizeof(unsigned int) * tile_n_x);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_idx_B_dev.at(device), sizeof(unsigned int) * tile_n_y);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&QT_dev.at(device), sizeof(double) * tile_n_x);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&means_A.at(device), sizeof(double) * tile_n_x);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&means_B.at(device), sizeof(double) * tile_n_y);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&norms_A.at(device), sizeof(double) * tile_n_x);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&norms_B.at(device), sizeof(double) * tile_n_y);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&df_A.at(device), sizeof(double) * tile_n_x);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&df_B.at(device), sizeof(double) * tile_n_y);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&dg_A.at(device), sizeof(double) * tile_n_x);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&dg_B.at(device), sizeof(double) * tile_n_y);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_A_merged.at(device), sizeof(unsigned long long int) * tile_n_x);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_B_merged.at(device), sizeof(unsigned long long int) * tile_n_y);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&scratchpad.at(device), sizeof(double) * tile_size);
        scratch[device] = new fft_precompute_helper(tile_size, m, true);
        cudaEvent_t st, ed, copy;
        cudaEventCreate(&ed);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventCreate(&st);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventCreate(&copy);
        gpuErrchk(cudaPeekAtLastError());
        
        clocks_start.emplace(device, st);
        clocks_end.emplace(device, ed);
        copy_to_host_done.emplace(device, copy);
        cudaStream_t s;
        cudaStreamCreate(&s);
        gpuErrchk(cudaPeekAtLastError());
        streams.emplace(device, s);
    }
    return SCAMP_NO_ERROR;

}

SCAMPError_t SCAMP_Operation::destroy()
{
    for (auto device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(T_A_dev[device]);
        cudaFree(T_B_dev[device]);
        cudaFree(QT_dev[device]);
        cudaFree(means_A[device]);
        cudaFree(means_B[device]);
        cudaFree(norms_A[device]);
        cudaFree(norms_B[device]);
        cudaFree(df_A[device]);
        cudaFree(df_B[device]);
        cudaFree(dg_A[device]);
        cudaFree(dg_B[device]);
        cudaFree(profile_A_dev[device]);
        cudaFree(profile_B_dev[device]);
        cudaFree(profile_A_merged[device]);
        cudaFree(profile_B_merged[device]);
        cudaFree(profile_idx_A_dev[device]);
        cudaFree(profile_idx_B_dev[device]);
        cudaFree(scratchpad.at(device));
        delete scratch[device];
        cudaEventDestroy(clocks_start[device]);
        cudaEventDestroy(clocks_end[device]);
        cudaEventDestroy(copy_to_host_done[device]);
        cudaStreamDestroy(streams.at(device));
    }
    return SCAMP_NO_ERROR;

}

SCAMPError_t SCAMP_Operation::do_tile(SCAMPTileType t, int device, const vector<double> &Ta_h, const vector<double> &Tb_h,
                                        const vector<float> &profile_h, const vector<unsigned int> &profile_idx_h,
                                        const vector<float> &profile_B_h, const vector<unsigned int> &profile_idx_B_h)
{
        size_t start_x = pos_x[device];
        size_t start_y = pos_y[device];
        MPIDXCombine combiner;
        SCAMPError_t err;
        size_t t_n_x = n_x[device] - m + 1;
        size_t t_n_y = n_y[device] - m + 1;
        printf("tile type = %d start_pos = [%lu, %lu]...\n", t, start_y, start_x);
        cudaMemcpyAsync(T_A_dev[device], Ta_h.data() + start_x, sizeof(double) * n_x[device], cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(T_B_dev[device], Tb_h.data() + start_y, sizeof(double) * n_y[device], cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_A_dev[device], profile_h.data() + start_x, sizeof(float) * t_n_x, cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_idx_A_dev[device], profile_idx_h.data() + start_x, sizeof(unsigned int) * t_n_x, cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        if(self_join) {
            cudaMemcpyAsync(profile_B_dev[device], profile_h.data() + start_y, sizeof(float) * t_n_y, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpyAsync(profile_idx_B_dev[device], profile_idx_h.data() + start_y, sizeof(unsigned int) * t_n_y, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
        } else if(full_join) {
            cudaMemcpyAsync(profile_B_dev[device], profile_B_h.data() + start_y, sizeof(float) * t_n_y, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            printf("start = %lu, size = %lu\n", start_y, profile_idx_B_h.size());
            cudaMemcpyAsync(profile_idx_B_dev[device], profile_idx_B_h.data() + start_y, sizeof(unsigned int) * t_n_y, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
        }
        // FIXME?: Computing the sliding dot products & statistics for each tile is overkill
        compute_statistics(T_A_dev[device], norms_A[device], df_A[device], dg_A[device], means_A[device], t_n_x, m, streams.at(device), scratchpad[device]);
        gpuErrchk(cudaPeekAtLastError());
        compute_statistics(T_B_dev[device], norms_B[device], df_B[device], dg_B[device],  means_B[device], t_n_y, m, streams.at(device), scratchpad[device]);
        gpuErrchk(cudaPeekAtLastError());
        thrust::device_ptr<unsigned long long int> ptr_A = thrust::device_pointer_cast(profile_A_merged[device]);
        thrust::transform(thrust::cuda::par.on(streams.at(device)), profile_A_dev[device], profile_A_dev[device] + t_n_x, profile_idx_A_dev[device], profile_A_merged[device], combiner);
        gpuErrchk(cudaPeekAtLastError());
        if(self_join || full_join) {
            thrust::device_ptr<unsigned long long int> ptr_B = thrust::device_pointer_cast(profile_B_merged[device]);
            thrust::transform(thrust::cuda::par.on(streams.at(device)), profile_B_dev[device], profile_B_dev[device] + t_n_y, profile_idx_B_dev[device], profile_B_merged[device], combiner);
            gpuErrchk(cudaPeekAtLastError());
        }
        SCAMP_Tile tile(t, T_A_dev[device], T_B_dev[device], df_A[device], df_B[device], dg_A[device], dg_B[device],
                         norms_A[device], norms_B[device], means_A[device], means_B[device],  QT_dev[device],
                         profile_A_merged[device], profile_B_merged[device], start_x, start_y, tile_start_col_position,
                         tile_start_row_position, n_y[device], n_x[device], m, scratch[device], dev_props.at(device), fp_type);
        cudaEventRecord(clocks_start[device], streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        err = tile.execute(streams.at(device));
        cudaEventRecord(clocks_end[device], streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        return err; 

}

void SCAMP_Operation::get_tile_ordering() {
    tile_ordering.clear();
    size_t num_tile_rows = ceil((size_B - m + 1) / (float) tile_n_y);
    size_t num_tile_cols = ceil((size_A - m + 1) / (float) tile_n_x);

    if(self_join) {
        for(int offset = 0; offset < num_tile_rows - 1; ++offset) {
            for(int diag = 0; diag < num_tile_cols - 1 - offset; ++diag) {
                tile_ordering.emplace_back(diag,diag + offset);
            }
        }

        for(int i = 0; i < num_tile_rows; ++i) {
            tile_ordering.emplace_back(i, num_tile_cols - 1);
        }
    } else {
        // Add upper diagonals one at a time except for edge tiles
        for(int diag = 0; diag < num_tile_cols - 1; ++diag) {
            for(int offset = 0; offset + diag < num_tile_cols - 1 && offset < num_tile_rows - 1; ++offset) {
                    tile_ordering.emplace_back(offset,diag + offset);
            }
        }
        
        // Add lower diagonals one at a time except for edge tiles
        for(int diag = 1; diag < num_tile_rows - 1; ++diag) {
            for(int offset = 0; offset + diag < num_tile_rows - 1 && offset < num_tile_cols - 1; ++offset) {
                tile_ordering.emplace_back(offset+diag,offset); 
            }
        }
    
        // Add the corner edge tile
        tile_ordering.emplace_back(num_tile_rows - 1, num_tile_cols - 1);

        int x = 0;
        int y = 0;

        // Alternate between adding final row and final column edge tiles
        while (x < num_tile_cols - 1 && y < num_tile_rows - 1) {
            tile_ordering.emplace_back(y, num_tile_cols - 1);
            tile_ordering.emplace_back(num_tile_rows - 1, x);
            ++x;
            ++y;
        }

        // Add any remaining final row edge tiles
        while (x < num_tile_cols - 1){
            tile_ordering.emplace_back(num_tile_rows - 1,x);
            ++x;
        }
    
        // Add any remaining final column edge tiles
        while (y < num_tile_rows - 1) {
            tile_ordering.emplace_back(y, num_tile_cols - 1);
            ++y;
        }
        
    }
    total_tiles = tile_ordering.size();
}


bool SCAMP_Operation::pick_and_start_next_tile(int dev, const vector<double> &Ta_h, const vector<double> &Tb_h,
                                                const vector<float> &profile_h, const vector<unsigned int> &profile_idx_h,
                                                const vector<float> &profile_B_h, const vector<unsigned int> &profile_idx_B_h)
{
    
    bool done = false;
    int tile_row = tile_ordering.front().first;
    int tile_col = tile_ordering.front().second;
    pos_x[dev] = tile_col * tile_n_x;
    pos_y[dev] = tile_row * tile_n_y;
    n_x[dev] = min(tile_size, size_A - pos_x[dev]);
    n_y[dev] = min(tile_size, size_B - pos_y[dev]);
    SCAMPError_t err;
    if(self_join) {
        if(tile_row == tile_col) {
            //partial tile on diagonal
            err = do_tile(SELF_JOIN_UPPER_TRIANGULAR, dev, Ta_h, Tb_h, profile_h, profile_idx_h, profile_B_h, profile_idx_B_h);
        } else {
            // full tile
            err = do_tile(SELF_JOIN_FULL_TILE, dev, Ta_h, Tb_h, profile_h, profile_idx_h, profile_B_h, profile_idx_B_h);
        }
    } else if(full_join) {
        err = do_tile(AB_FULL_JOIN_FULL_TILE, dev, Ta_h, Tb_h, profile_h, profile_idx_h, profile_B_h, profile_idx_B_h);
    } else {
        err = do_tile(AB_JOIN_FULL_TILE, dev, Ta_h, Tb_h, profile_h, profile_idx_h, profile_B_h, profile_idx_B_h);
    }
    if (err != SCAMP_NO_ERROR) {
        printf("ERROR %d executing tile. \n", err);
    }
    tile_ordering.pop_front();
    if(tile_ordering.empty()){
        done = true;
    }
    return done;
}


void merge_partial_on_host(vector<unsigned long long int> &profile_to_merge, vector<float> &merge_target, vector<unsigned int> &merge_idx_target, size_t merge_start, size_t tile_sz)
{
    auto iter_begin = thrust::make_zip_iterator(thrust::make_tuple(merge_target.data() + merge_start, merge_idx_target.data() + merge_start, profile_to_merge.data()));
    auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(merge_target.data() + merge_start + tile_sz, merge_idx_target.data() + merge_start + tile_sz, profile_to_merge.data() + tile_sz));
    thrust::for_each(iter_begin, iter_end, max_with_index());

}


int SCAMP_Operation::issue_and_merge_tiles_on_devices(const vector<double> &Ta_host, const vector<double> &Tb_host,
                                                       vector<float> &profile_A_full_host, vector<unsigned int> &profile_idx_A_full_host,
                                                       vector<float> &profile_B_full_host, vector<unsigned int> &profile_idx_B_full_host,
                                                       vector<vector<unsigned long long int>> &profileA_h,
                                                       vector<vector<unsigned long long int>> &profileB_h,
                                                       int last_device_idx = ISSUED_ALL_DEVICES)
{
    bool done = last_device_idx != ISSUED_ALL_DEVICES;
    int last_dev = ISSUED_ALL_DEVICES; 
    if(last_device_idx == ISSUED_ALL_DEVICES) {
        last_device_idx = devices.size() - 1;
    }
    for(int i = 0; i <= last_device_idx; ++i) {
        int device = devices.at(i);
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profileA_h.at(i).data(), profile_A_merged[device], sizeof(unsigned long long int) * (n_x[device] - m + 1), cudaMemcpyDeviceToHost, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        if(self_join || full_join) {
            cudaMemcpyAsync(profileB_h.at(i).data(), profile_B_merged[device], sizeof(unsigned long long int) * (n_y[device] - m + 1), cudaMemcpyDeviceToHost, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
        }
        cudaEventRecord(copy_to_host_done[device], streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        n_x_2[device] = n_x[device];
        n_y_2[device] = n_y[device];
        pos_x_2[device] = pos_x[device];
        pos_y_2[device] = pos_y[device];
        if(!done) {
            done = pick_and_start_next_tile(device, Ta_host, Tb_host, profile_A_full_host, profile_idx_A_full_host, profile_B_full_host, profile_idx_B_full_host);
            if(done) {
                last_dev = i;
            }
        }
    }

    for(int i = 0; i <= last_device_idx; ++i) {
        int device = devices.at(i);
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventSynchronize(copy_to_host_done[device]);
        gpuErrchk(cudaPeekAtLastError());
        merge_partial_on_host(profileA_h.at(i), profile_A_full_host, profile_idx_A_full_host, pos_x_2[device], (n_x_2[device] - m + 1));
        gpuErrchk(cudaPeekAtLastError());
        if(self_join) {
            merge_partial_on_host(profileB_h.at(i), profile_A_full_host, profile_idx_A_full_host, pos_y_2[device], (n_y_2[device] - m + 1));
            gpuErrchk(cudaPeekAtLastError());
        } else if(full_join) {
            merge_partial_on_host(profileB_h.at(i), profile_B_full_host, profile_idx_B_full_host, pos_y_2[device], (n_y_2[device] - m + 1));
            gpuErrchk(cudaPeekAtLastError());
        }
        completed_tiles++;
        printf("%f percent complete\n", (completed_tiles / (float) total_tiles) * 100);
    }
    return last_dev;
}


SCAMPError_t SCAMP_Operation::do_join(const vector<double> &Ta_host, const vector<double> &Tb_host, vector<float> &profile, vector<unsigned int> &profile_idx, vector<float> &profile_B, vector<unsigned int> &profile_idx_B)
{

    vector< vector<unsigned long long int> > profileA_h(devices.size(), vector<unsigned long long int>(tile_n_y)), profileB_h(devices.size(), vector<unsigned long long int>(tile_n_x));
    bool done = false;
    int last_dev = ISSUED_ALL_DEVICES;
    get_tile_ordering();
    printf("Performing join with %lu tiles.\n", tile_ordering.size() );
    for(int i = 0; i < devices.size(); ++i) {
        int device = devices.at(i);
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        done = pick_and_start_next_tile(device, Ta_host, Tb_host, profile, profile_idx, profile_B, profile_idx_B);
        gpuErrchk(cudaPeekAtLastError());
        if (done) {
            last_dev = i;
            break;
        }
    }

    while(last_dev == ISSUED_ALL_DEVICES) {
        last_dev = issue_and_merge_tiles_on_devices(Ta_host, Tb_host, profile, profile_idx, profile_B, profile_idx_B, profileA_h, profileB_h);
    }

    issue_and_merge_tiles_on_devices(Ta_host, Tb_host, profile, profile_idx, profile_B, profile_idx_B, profileA_h, profileB_h, last_dev);

    return SCAMP_NO_ERROR;
}

void do_SCAMP(const vector<double> &Ta_h, const vector<double> &Tb_h, vector<float> &profile_h, vector<unsigned int> &profile_idx_h,
               vector<float> &profile_B_h, vector<unsigned int> &profile_idx_B_h, const unsigned int m, const size_t max_tile_size, 
               const vector<int> &devices, bool self_join, FPtype t, bool full_join, size_t start_row, size_t start_col)
{
    if(devices.empty()) {
        printf("Error: no gpu provided\n");
        exit(0);
    }
    // Allocate and initialize memory
    clock_t start, end;
    SCAMP_Operation op(Ta_h.size(), Tb_h.size(), m, max_tile_size, devices, self_join, t, full_join, start_row, start_col);
    op.init();
    gpuErrchk(cudaPeekAtLastError());
    start = clock();
    if(self_join) {
        op.do_join(Ta_h, Ta_h, profile_h, profile_idx_h, profile_B_h, profile_idx_B_h);
    } else {
        op.do_join(Ta_h, Tb_h, profile_h, profile_idx_h, profile_B_h, profile_idx_B_h);

    }
    cudaDeviceSynchronize();
    end = clock();
    gpuErrchk(cudaPeekAtLastError());
    op.destroy();
    gpuErrchk(cudaPeekAtLastError());
    printf("Finished SCAMP to generate partial matrix profile of size %lu in %f seconds on %lu devices:\n", profile_h.size(), (end - start) / (double) CLOCKS_PER_SEC, devices.size());
}

//Reads input time series from file
template<class DTYPE>
void readFile(const char* filename, vector<DTYPE>& v, const char *format_str) 
{
    FILE* f = fopen( filename, "r");
    if(f == NULL){
        printf("Unable to open %s for reading, please make sure it exists\n", filename);
        exit(0);
    }
    DTYPE num;
    while(!feof(f)){
            fscanf(f, format_str, &num);
            v.push_back(num);
        }
    v.pop_back();
    fclose(f);
}
    
}

int main(int argc, char** argv) {
    SCAMP::FPtype t = SCAMP::FP_SINGLE;
    bool full_join = false;
    bool self_join = true;
    size_t start_row = 0;
    size_t start_col = 0;
    int max_tile_size = (1 << 20);
    int opt;
    vector<int> devices;
    vector<double> Ta_h, Tb_h;
    char *output_B_prefix, *input_B;
    while ((opt = getopt(argc, argv, "mdf:r:c:s:b:g:")) != -1) {
        switch (opt) {
        case 'd': t = SCAMP::FP_DOUBLE; break;
        case 'm': t = SCAMP::FP_MIXED; break;
        case 'f': output_B_prefix = optarg; full_join = true; break;
        case 'r': start_row = atoi(optarg); break;
        case 'c': start_col = atoi(optarg); break;
        case 's': max_tile_size = atoi(optarg); break;
        case 'b': input_B = optarg; self_join = false; break;
        case 'g': devices.push_back(atoi(optarg)); break;
        default:
            exit(EXIT_FAILURE);
        }
    }
    if(self_join && full_join) {
        printf("error: invalid argument combination -f flag can only be used in ab-joins");
        exit(EXIT_FAILURE);
    }

    int index = optind;
    int window_size = atoi(argv[index++]);
    char *input_A = argv[index++];
     
    SCAMP::readFile<double>(input_A, Ta_h, "%lf");
    
    if(!self_join) {
        SCAMP::readFile<double>(input_B, Tb_h, "%lf"); 
    }

    int n_x = Ta_h.size() - window_size + 1;
    int n_y;
    if(self_join) {
       n_y = n_x;
    } else {
       n_y = Tb_h.size() - window_size + 1;
    }

    vector<float> profile(n_x, CC_MIN);
    vector<unsigned int> profile_idx(n_x, 0);
    vector<float> profile_B;
    vector<unsigned int> profile_idx_B;

    if(full_join){
        profile_B = vector<float>(n_y, CC_MIN);
        profile_idx_B = vector<unsigned int>(n_y, 0);
    }

    cudaFree(0);
    
    if(devices.empty()) {
        // Use all available devices
        printf("using all devices\n"); 
        int num_dev;
        cudaGetDeviceCount(&num_dev);
        for(int i = 0; i < num_dev; ++i){ 
            devices.push_back(i);
        }
    }

    printf("Starting SCAMP\n");     
    SCAMP::do_SCAMP(Ta_h, Tb_h, profile, profile_idx, profile_B, profile_idx_B, window_size, max_tile_size, devices, self_join, t, full_join, start_row, start_col);
    
    printf("Now writing result to files\n");
    FILE* f1 = fopen( argv[index++], "w");
    FILE* f2 = fopen( argv[index++], "w");
    FILE* f3, *f4;
    for(int i = 0; i < profile.size(); ++i){
         fprintf(f1, "%f\n", sqrt(max(2*window_size*(1 - profile[i]), 0.0)));
         fprintf(f2, "%u\n", profile_idx[i] + 1);
    }
    fclose(f1);
    fclose(f2);
    if(full_join) {
        f3 = fopen(strcat(output_B_prefix,"_mp") , "w");
        f4 = fopen(strcat(output_B_prefix,"i"), "w");
        for(int i = 0; i < profile_B.size(); ++i) {
            fprintf(f3, "%f\n", sqrt(max(2*window_size*(1 - profile_B[i]), 0.0)));
            fprintf(f4, "%u\n", profile_idx_B[i] + 1);
        }
    }
    if(full_join) {
        fclose(f3);
        fclose(f4);
    }
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaDeviceReset());
    printf("Done\n");
    return 0;
}

