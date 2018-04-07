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
#include "SCRIMP.h"
#include "tile.h"

using std::vector;
using std::unordered_map;
using std::make_pair;

#define WORK_SIZE 512
#define AMT_UNROLL 16
namespace SCRIMP {

__global__ void cross_correlation_to_ed(float *profile, unsigned int n, unsigned int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        profile[tid] = sqrt(max(2*(m - profile[tid]), 0.0));
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

// This kernel computes the recipricol sliding standard deviaiton with specified window size, the corresponding means of each element, and the prefix squared sum at each element
// We actually compute the multiplicative inverse of the standard deviation, as this saves us from needing to do a division in the main kernel
template<class DTYPE>
__global__ void sliding_std(DTYPE* cumsumsqr, unsigned int window, unsigned int size, DTYPE* means, DTYPE* stds) {
    const DTYPE coeff = 1 / (DTYPE) window;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x * blockDim.x + threadIdx.x + window;
    if (a == 0) {
        stds[a] = 1 / sqrt((cumsumsqr[window - 1] * coeff) - (means[a] * means[a]));
    }
    else if (b < size + window) {
        stds[a] = 1 / sqrt(((cumsumsqr[b - 1] - cumsumsqr[a - 1]) * coeff) - (means[a] * means[a]));
    }
}

template<class DTYPE>
void compute_statistics(const DTYPE *T, DTYPE *means, DTYPE *stds, size_t n, size_t m, cudaStream_t s, DTYPE *scratch)
{
    square<DTYPE> sqr;
    dim3 grid(ceil(n / (double) WORK_SIZE), 1,1);
    dim3 block(WORK_SIZE, 1, 1);
    
    gpuErrchk(cudaPeekAtLastError());
    
    thrust::device_ptr<const DTYPE> dev_ptr_T = thrust::device_pointer_cast(T);
    thrust::device_ptr<DTYPE> dev_ptr_scratch = thrust::device_pointer_cast(scratch);

    // Compute prefix sum in scratch
    thrust::inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum to compute sliding mean
    sliding_mean<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, means);
    gpuErrchk(cudaPeekAtLastError());
    // Compute prefix sum of squares in scratch
    thrust::transform_inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, sqr,thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum of squares to compute the sliding standard deviation
    sliding_std<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, means, stds);
    gpuErrchk(cudaPeekAtLastError());
}

template<class DTYPE, class CUFFT_DTYPE>
SCRIMPError_t SCRIMP_Operation<DTYPE,CUFFT_DTYPE>::init()
{
    for (auto device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        
        T_A_dev.insert(make_pair(device, (DTYPE*) 0));
        T_B_dev.insert(make_pair(device, (DTYPE*) 0));
        QT_dev.insert(make_pair(device, (DTYPE*) 0));
        means_A.insert(make_pair(device, (DTYPE*) 0));
        means_B.insert(make_pair(device, (DTYPE*) 0));
        stds_A.insert(make_pair(device, (DTYPE*) 0));
        stds_B.insert(make_pair(device, (DTYPE*) 0));
        profile_A_dev.insert(make_pair(device,(float*) NULL));
        profile_B_dev.insert(make_pair(device,(float*) NULL));
        profile_A_merged.insert(make_pair(device,(unsigned long long int*) NULL));
        profile_B_merged.insert(make_pair(device,(unsigned long long int*) NULL));
        profile_idx_A_dev.insert(make_pair(device,(unsigned int *) NULL));
        profile_idx_B_dev.insert(make_pair(device,(unsigned int *) NULL));
        scratchpad.insert(make_pair(device, (DTYPE*) NULL));

        cudaMalloc(&T_A_dev.at(device), sizeof(DTYPE) * tile_size);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&T_B_dev.at(device), sizeof(DTYPE) * tile_size);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_A_dev.at(device), sizeof(float) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_B_dev.at(device), sizeof(float) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_idx_A_dev.at(device), sizeof(unsigned int) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_idx_B_dev.at(device), sizeof(unsigned int) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&QT_dev.at(device), sizeof(DTYPE) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&means_A.at(device), sizeof(DTYPE) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&means_B.at(device), sizeof(DTYPE) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&stds_A.at(device), sizeof(DTYPE) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&stds_B.at(device), sizeof(DTYPE) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_A_merged.at(device), sizeof(unsigned long long int) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_B_merged.at(device), sizeof(unsigned long long int) * tile_n);
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&scratchpad.at(device), sizeof(DTYPE) * tile_size);
        scratch[device] = new fft_precompute_helper<DTYPE, CUFFT_DTYPE>(tile_size, m, true);
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
    return SCRIMP_NO_ERROR;

}

template<class DTYPE, class CUFFT_DTYPE>
SCRIMPError_t SCRIMP_Operation<DTYPE,CUFFT_DTYPE>::destroy()
{
    for (auto device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(T_A_dev[device]);
        cudaFree(T_B_dev[device]);
        cudaFree(QT_dev[device]);
        cudaFree(means_A[device]);
        cudaFree(means_B[device]);
        cudaFree(stds_A[device]);
        cudaFree(stds_B[device]);
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
    return SCRIMP_NO_ERROR;

}

template<class DTYPE, class CUFFT_DTYPE>
SCRIMPError_t SCRIMP_Operation<DTYPE,CUFFT_DTYPE>::do_tile(SCRIMPTileType t, size_t t_size_x, size_t t_size_y, size_t start_x, size_t start_y, int device, const vector<DTYPE> &T_h, const vector<float> &profile_h, const vector<unsigned int> &profile_idx_h)
{ 
        MPIDXCombine combiner;
        SCRIMPError_t err;
        size_t t_n_x = t_size_x - m + 1;
        size_t t_n_y = t_size_y - m + 1;
        printf("tile type = %d start_pos = [%lu, %lu]...\n", t, start_x, start_y);
        cudaMemcpyAsync(T_A_dev[device], T_h.data() + start_x, sizeof(DTYPE) * t_size_x, cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(T_B_dev[device], T_h.data() + start_y, sizeof(DTYPE) * t_size_y, cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_A_dev[device], profile_h.data() + start_x, sizeof(float) * t_n_x, cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_B_dev[device], profile_h.data() + start_y, sizeof(float) * t_n_y, cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_idx_A_dev[device], profile_idx_h.data() + start_x, sizeof(unsigned int) * t_n_x, cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_idx_B_dev[device], profile_idx_h.data() + start_y, sizeof(unsigned int) * t_n_y, cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        // TODO: Computing the sliding dot products & statistics for each tile is overkill
        compute_statistics<DTYPE>(T_A_dev[device], means_A[device], stds_A[device], t_n_x, m, streams.at(device), scratchpad[device]);
        gpuErrchk(cudaPeekAtLastError());
        compute_statistics<DTYPE>(T_B_dev[device], means_B[device], stds_B[device], t_n_y, m, streams.at(device), scratchpad[device]);
        gpuErrchk(cudaPeekAtLastError());
        thrust::device_ptr<unsigned long long int> ptr_A = thrust::device_pointer_cast(profile_A_merged[device]);
        thrust::device_ptr<unsigned long long int> ptr_B = thrust::device_pointer_cast(profile_B_merged[device]);
        thrust::transform(thrust::cuda::par.on(streams.at(device)), profile_A_dev[device], profile_A_dev[device] + t_n_x, profile_idx_A_dev[device], profile_A_merged[device], combiner);
        gpuErrchk(cudaPeekAtLastError());
        thrust::transform(thrust::cuda::par.on(streams.at(device)), profile_B_dev[device], profile_B_dev[device] + t_n_y, profile_idx_B_dev[device], profile_B_merged[device], combiner);
        gpuErrchk(cudaPeekAtLastError());

        SCRIMP_Tile<DTYPE, CUFFT_DTYPE, WORK_SIZE, AMT_UNROLL> tile = SCRIMP_Tile<DTYPE, CUFFT_DTYPE, WORK_SIZE, AMT_UNROLL>(t, T_A_dev[device], T_B_dev[device], means_A[device], means_B[device], stds_A[device], stds_B[device], QT_dev[device], profile_A_merged[device], profile_B_merged[device], start_x, start_y, t_size_y, t_size_x, m, scratch[device]);
        cudaEventRecord(clocks_start[device], streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        err = tile.execute(streams.at(device));
        cudaEventRecord(clocks_end[device], streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        return err; 

}


template<class DTYPE, class CUFFT_DTYPE>
bool SCRIMP_Operation<DTYPE,CUFFT_DTYPE>::pick_and_start_next_tile_self_join(int dev, size_t &tile_col, size_t &work_start, vector<int> &curr_tile, const size_t num_tile_rows, const size_t num_tile_cols, const vector<DTYPE> &T_h, const vector<float> &profile_h, const vector<unsigned int> &profile_idx_h, size_t &size_x, size_t &size_y, size_t &start_x, size_t &start_y)
{
    
    bool done = false;
    int tile_row = curr_tile.at(tile_col);
    start_x = tile_col * tile_n;
    start_y = tile_row * tile_n;
    size_x = min(tile_size, size_A - start_x);
    size_y = min(tile_size, size_B - start_y);
    printf("start_x = %lu, start_y = %lu, size_x = %lu, size_y = %lu, tile_size = %lu\n", start_x, start_y, size_x, size_y, tile_size);
    if(tile_row == tile_col) {
        //partial tile on diagonal
        do_tile(SELF_JOIN_UPPER_TRIANGULAR, size_x, size_y, start_x, start_y, dev, T_h, profile_h, profile_idx_h);
        curr_tile.at(tile_col)--;
        if(curr_tile.at(tile_col) < 0) {
            work_start++;
        }
    } else {
        // full tile
        do_tile(SELF_JOIN_FULL_TILE, size_x, size_y, start_x, start_y, dev, T_h, profile_h, profile_idx_h);
        curr_tile.at(tile_col)--;
        if(curr_tile.at(tile_col) < 0) {
            work_start++;
        }
    }
    if(work_start >=  num_tile_rows){ 
        done = true;
    }
    tile_col++;
    if(tile_col >= num_tile_cols) {
        tile_col = work_start;
    }
    return done;
}


void merge_partial_on_host(vector<unsigned long long int> &profile_to_merge, vector<float> &merge_target, vector<unsigned int> &merge_idx_target, size_t merge_start, size_t tile_sz)
{
    auto iter_begin = thrust::make_zip_iterator(thrust::make_tuple(merge_target.data() + merge_start, merge_idx_target.data() + merge_start, profile_to_merge.data()));
    auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(merge_target.data() + merge_start + tile_sz, merge_idx_target.data() + merge_start + tile_sz, profile_to_merge.data() + tile_sz));
    thrust::for_each(iter_begin, iter_end, max_with_index());

}

template<class DTYPE, class CUFFT_DTYPE>
SCRIMPError_t SCRIMP_Operation<DTYPE, CUFFT_DTYPE>::do_self_join(const vector<DTYPE> &T_host, vector<float> &profile, vector<unsigned int> &profile_idx)
{
    
    size_t num_tile_rows = ceil((size_A - m + 1) / (float) tile_n);
    size_t num_tile_cols = ceil((size_B - m + 1) / (float) tile_n);
    vector<bool> profile_lock(num_tile_cols, false);
    vector<int> curr_tile(num_tile_rows);
    vector< vector<unsigned long long int> > profileA_h(devices.size(), vector<unsigned long long int>(tile_n)), profileB_h(devices.size(), vector<unsigned long long int>(tile_n));
    std::iota(curr_tile.begin(), curr_tile.end(), 0);
    bool done = false;
    size_t tile_col = 0;
    size_t work_start = 0;
    int last_dev;
    vector<size_t> n_x(devices.size());
    vector<size_t> n_y(devices.size());
    vector<size_t> n_x_2(devices.size());
    vector<size_t> n_y_2(devices.size());
    vector<size_t> pos_x(devices.size());
    vector<size_t> pos_y(devices.size());
    vector<size_t> pos_x_2(devices.size());
    vector<size_t> pos_y_2(devices.size());
    

    for(auto device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        done = pick_and_start_next_tile_self_join(device, tile_col, work_start, curr_tile, num_tile_rows, num_tile_cols, T_host, profile, profile_idx, n_x[device], n_y[device], pos_x[device], pos_y[device]);
        gpuErrchk(cudaPeekAtLastError());
        if (done) {
            last_dev = device;
            break;
        }
    }

    while(!done) {
        for(auto device : devices) {
            cudaSetDevice(device);
            gpuErrchk(cudaPeekAtLastError());
            cudaEventSynchronize(clocks_end[device]);
            gpuErrchk(cudaPeekAtLastError());
            float elapsed_time;
            cudaEventElapsedTime(&elapsed_time, clocks_start[device], clocks_end[device]);
            gpuErrchk(cudaPeekAtLastError());
            printf("Device %d took %f seconds to finish tile\n", device, elapsed_time / 1000);
            cudaMemcpyAsync(profileA_h.at(device).data(), profile_A_merged[device], sizeof(unsigned long long int) * (n_x[device] - m + 1), cudaMemcpyDeviceToHost, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpyAsync(profileB_h.at(device).data(), profile_B_merged[device], sizeof(unsigned long long int) * (n_y[device] - m + 1), cudaMemcpyDeviceToHost, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            cudaEventRecord(copy_to_host_done[device], streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            n_x_2[device] = n_x[device];
            n_y_2[device] = n_y[device];
            pos_x_2[device] = pos_x[device];
            pos_y_2[device] = pos_y[device];
            if(!done) {
                done = pick_and_start_next_tile_self_join(device, tile_col, work_start, curr_tile, num_tile_rows, num_tile_cols, T_host, profile, profile_idx, n_x[device], n_y[device], pos_x[device], pos_y[device]);
                if(done) {
                    last_dev = device;
                }
            }
        }

        for(auto device : devices) {
            cudaSetDevice(device);
            gpuErrchk(cudaPeekAtLastError());
            cudaEventSynchronize(copy_to_host_done[device]);
            gpuErrchk(cudaPeekAtLastError());
            merge_partial_on_host(profileA_h.at(device), profile, profile_idx, pos_x_2[device], (n_x_2[device] - m + 1));
            gpuErrchk(cudaPeekAtLastError());
            merge_partial_on_host(profileB_h.at(device), profile, profile_idx, pos_y_2[device], (n_y_2[device] - m + 1));
            gpuErrchk(cudaPeekAtLastError());
            
        }


    }

    for(int device = 0; device <= last_dev; ++device) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventSynchronize(clocks_end[device]);
        gpuErrchk(cudaPeekAtLastError());
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, clocks_start[device], clocks_end[device]);
        gpuErrchk(cudaPeekAtLastError());
        printf("Device %d took %f seconds to finish tile\n", device, elapsed_time / 1000);
        cudaMemcpyAsync(profileA_h.at(device).data(), profile_A_merged[device], sizeof(unsigned long long int) * (n_x[device] - m + 1), cudaMemcpyDeviceToHost, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profileB_h.at(device).data(), profile_B_merged[device], sizeof(unsigned long long int) * (n_y[device] - m + 1), cudaMemcpyDeviceToHost, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaEventRecord(copy_to_host_done[device], streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
    }
    for(int device = 0; device <= last_dev; ++device) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventSynchronize(copy_to_host_done[device]);
        gpuErrchk(cudaPeekAtLastError());
        merge_partial_on_host(profileA_h.at(device), profile, profile_idx, pos_x[device], (n_x[device] - m + 1));
        gpuErrchk(cudaPeekAtLastError());
        merge_partial_on_host(profileB_h.at(device), profile, profile_idx, pos_y[device], (n_y[device] - m + 1));
        gpuErrchk(cudaPeekAtLastError());
    }
    return SCRIMP_NO_ERROR;
}

template<class DTYPE, class CUFFT_DTYPE>
void do_SCRIMP(const vector<DTYPE> &T_h, vector<float> &profile_h, vector<unsigned int> &profile_idx_h, const unsigned int m, const vector<int> &devices) {
    if(devices.empty()) {
        printf("Error: no gpu provided\n");
        exit(0);
    }
    // Allocate and initialize memory
    clock_t start, end;
    SCRIMP_Operation<DTYPE, CUFFT_DTYPE> op(T_h.size(), T_h.size(), m, devices);
    op.init();
    gpuErrchk(cudaPeekAtLastError());
    start = clock();
    op.do_self_join(T_h, profile_h, profile_idx_h);
    cudaDeviceSynchronize();
    end = clock();
    gpuErrchk(cudaPeekAtLastError());
    op.destroy();
    gpuErrchk(cudaPeekAtLastError());
    printf("Finished STOMP to generate partial matrix profile of size %lu in %f seconds on %lu devices:\n", profile_h.size(), (end - start) / (double) CLOCKS_PER_SEC, devices.size());
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

    if(argc < 5) {
        printf("Usage: SCRIMP <window_len> <input file> <profile output file> <index output file> [Optional: list of GPU device numbers to run on]\n");
        exit(0);
    }

    int window_size = atoi(argv[1]);
    
    vector<double> T_h;
    SCRIMP::readFile<double>(argv[2], T_h, "%lf");
    int n = T_h.size() - window_size + 1;
    vector<float> profile(n, CC_MIN);
    vector<unsigned int> profile_idx(n, 0);
    
    cudaFree(0);
    
    vector<int> devices;
    
    if(argc == 5) {
        // Use all available devices 
        int num_dev;
        cudaGetDeviceCount(&num_dev);
        for(int i = 0; i < num_dev; ++i){ 
            devices.push_back(i);
        }
    } else {
        // Use the devices specified
        int x = 5;
        while (x < argc) {
            devices.push_back(atoi(argv[x]));
            ++x;
        }
    }
    
    printf("Starting SCRIMP\n");
     
    SCRIMP::do_SCRIMP<double, cuDoubleComplex>(T_h, profile, profile_idx, window_size, devices);
    
    printf("Now writing result to files\n");
    FILE* f1 = fopen( argv[3], "w");
    FILE* f2 = fopen( argv[4], "w");
    for(int i = 0; i < profile.size(); ++i){
         fprintf(f1, "%f\n", sqrt(max(2*(window_size - profile[i]), 0.0)));
         fprintf(f2, "%u\n", profile_idx[i] + 1);
    }
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaDeviceReset());
    fclose(f1);
    fclose(f2);
    printf("Done\n");
    return 0;
}

