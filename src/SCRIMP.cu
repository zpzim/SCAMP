#include <vector>
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
void compute_statistics(const DTYPE *T, DTYPE *means, DTYPE *stds, size_t n, size_t m, cudaStream_t s)
{
    square<DTYPE> sqr;
    dim3 grid(ceil(n / (double) WORK_SIZE), 1,1);
    dim3 block(WORK_SIZE, 1, 1);
    
    DTYPE *scratch;
    cudaMalloc(&scratch, sizeof(DTYPE) * (n + m - 1));
    gpuErrchk(cudaPeekAtLastError());
    
    thrust::device_ptr<const DTYPE> dev_ptr_T = thrust::device_pointer_cast(T);
    thrust::device_ptr<DTYPE> dev_ptr_scratch = thrust::device_pointer_cast(scratch);

    // Compute prefix sum in scratch
    thrust::inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    printf("hello\n");
    // Use prefix sum to compute sliding mean
    printf("size = %lu\n", n);
    sliding_mean<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, means);
    gpuErrchk(cudaPeekAtLastError());
    printf("hello\n");
    // Compute prefix sum of squares in scratch
    thrust::transform_inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, sqr,thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    printf("hello\n");
    // Use prefix sum of squares to compute the sliding standard deviation
    sliding_std<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, means, stds);
    gpuErrchk(cudaPeekAtLastError());
    cudaStreamSynchronize(s);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(scratch);
    gpuErrchk(cudaPeekAtLastError());
}

template<class DTYPE, class CUFFT_DTYPE>
void do_SCRIMP(const vector<DTYPE> &T_h, vector<float> &profile_h, vector<unsigned int> &profile_idx_h, const unsigned int m, const vector<int> &devices) {
    if(devices.empty()) {
        printf("Error: no gpu provided\n");
        exit(0);
    }
    size_t tile_size = T_h.size();
    size_t n = T_h.size() - m + 1;
    size_t tile_n = tile_size - m + 1;
 
    unordered_map<int, DTYPE*> T_A_dev, T_B_dev, QT_dev, means_A, means_B, stds_A, stds_B;
    unordered_map<int, float*> profile_A_dev, profile_B_dev;
    unordered_map<int, unsigned long long int*> profile_A_merged, profile_B_merged;
    unordered_map<int, unsigned int*> profile_idx_A_dev, profile_idx_B_dev;
    unordered_map<int, cudaEvent_t> clocks_start, clocks_end;
    unordered_map<int, cudaStream_t> streams;
    unordered_map<int, fft_precompute_helper<DTYPE, CUFFT_DTYPE>*> scratch;

    // Allocate and initialize memory
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
        scratch[device] = new fft_precompute_helper<DTYPE, CUFFT_DTYPE>(tile_size, m, true);
        cudaEvent_t st, ed;
        cudaEventCreate(&ed);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventCreate(&st);
        gpuErrchk(cudaPeekAtLastError());
        clocks_start.emplace(device, st);
        clocks_end.emplace(device, ed);
        cudaStream_t s;
        cudaStreamCreate(&s);
        gpuErrchk(cudaPeekAtLastError());
        streams.emplace(device, s);
    }

    MPIDXCombine combiner;
    
    // Asynchronously copy relevant data, precompute statistics, generate partial matrix profile
    int count = 0;
    for (int i = 0; i < n; i += tile_n) {
        for(int j = 0; j < n; j += tile_n) {
            printf("tile [%d, %d]...\n", i, j);
            int device = count % devices.size();
            cudaSetDevice(device);
            cudaMemcpyAsync(T_A_dev[device], T_h.data() + j, sizeof(DTYPE) * tile_size, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpyAsync(T_B_dev[device], T_h.data() + i, sizeof(DTYPE) * tile_size, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpyAsync(profile_A_dev[device], profile_h.data() + j, sizeof(float) * tile_n, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpyAsync(profile_B_dev[device], profile_h.data() + i, sizeof(float) * tile_n, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpyAsync(profile_idx_A_dev[device], profile_idx_h.data() + j, sizeof(unsigned int) * tile_n, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpyAsync(profile_idx_B_dev[device], profile_idx_h.data() + i, sizeof(unsigned int) * tile_n, cudaMemcpyHostToDevice, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            // TODO: Computing the sliding dot products & statistics for each tile is overkill
            compute_statistics<DTYPE>(T_A_dev[device], means_A[device], stds_A[device], tile_n, m, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            compute_statistics<DTYPE>(T_B_dev[device], means_B[device], stds_B[device], tile_n, m, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
            thrust::device_ptr<unsigned long long int> ptr_A = thrust::device_pointer_cast(profile_A_merged[device]);
            thrust::device_ptr<unsigned long long int> ptr_B = thrust::device_pointer_cast(profile_B_merged[device]);
            thrust::transform(thrust::cuda::par.on(streams.at(device)), profile_A_dev[device], profile_A_dev[device] + tile_n, profile_idx_A_dev[device], profile_A_merged[device], combiner);
            gpuErrchk(cudaPeekAtLastError());
            thrust::transform(thrust::cuda::par.on(streams.at(device)), profile_B_dev[device], profile_B_dev[device] + tile_n, profile_idx_B_dev[device], profile_B_merged[device], combiner);
            gpuErrchk(cudaPeekAtLastError());
            SCRIMP_Tile<DTYPE, CUFFT_DTYPE, WORK_SIZE, AMT_UNROLL> tile = SCRIMP_Tile<DTYPE, CUFFT_DTYPE, WORK_SIZE, AMT_UNROLL>(SELF_JOIN_UPPER_TRIANGULAR, T_A_dev[device], T_B_dev[device], means_A[device], means_B[device], stds_A[device], stds_B[device], QT_dev[device], profile_A_merged[device], profile_B_merged[device], j, i, tile_size, tile_size, m, scratch[device]);
            printf("Starting tile %d main kernel on GPU %d\n", count, device);
            cudaEventRecord(clocks_start[device], streams.at(device));
            if(tile.execute(streams.at(device)) != SCRIMP_NO_ERROR) {
                printf("problem with tile %d on device %d\n", count, device);
            }
            cudaEventRecord(clocks_end[device], streams.at(device));
            ++count;
        }
    }
    float time;
    for(auto &device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaStreamSynchronize(streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaEventElapsedTime(&time, clocks_start[device], clocks_end[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventDestroy(clocks_start.at(device));
        cudaEventDestroy(clocks_end.at(device));
        printf("Device %d took %f seconds\n", device, time / 1000);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    printf("Finished STOMP to generate partial matrix profile of size %lu on %lu devices:\n", n, devices.size());

    // Free unneeded resources
    for (auto &device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(T_A_dev[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(T_B_dev[device]);
        gpuErrchk(cudaPeekAtLastError());
        // Keep the profile for the first device as a staging area for the final result
        if (device != devices.at(0)) { 
            cudaFree(profile_A_dev[device]);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(profile_B_dev[device]);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(profile_idx_A_dev[device]);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(profile_idx_B_dev[device]);
            gpuErrchk(cudaPeekAtLastError());
        }
        cudaFree(QT_dev[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(means_A[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(means_B[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(stds_A[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(stds_B[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaStreamDestroy(streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        delete scratch[device];
    }
    
    cudaSetDevice(0);
    auto ptr_profile = thrust::device_ptr<float>(profile_A_dev[devices.at(0)]);
    auto ptr_profile_B = thrust::device_ptr<float>(profile_A_dev[devices.at(0)]);
    auto ptr_index = thrust::device_ptr<unsigned int>(profile_idx_A_dev[devices.at(0)]);
    auto ptr_index_B = thrust::device_ptr<unsigned int>(profile_idx_B_dev[devices.at(0)]);
    auto ptr_merged = thrust::device_ptr<unsigned long long int>(profile_A_merged[devices.at(0)]);
    auto ptr_merged_B = thrust::device_ptr<unsigned long long int>(profile_B_merged[devices.at(0)]);
    auto iter_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile, ptr_index, ptr_merged));
    auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile + tile_n, ptr_index + tile_n, ptr_merged + tile_n));
    thrust::for_each(iter_begin, iter_end, max_with_index());
    iter_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile, ptr_index, ptr_merged_B));
    iter_end = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile + tile_n, ptr_index + tile_n, ptr_merged_B + tile_n));
    thrust::for_each(iter_begin, iter_end, max_with_index());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

/*
    // Consolidate the partial matrix profiles to a single vector using the first gpu 
    printf("Merging partial matrix profiles into final result\n");
    vector<unsigned long long int> partial_profile_host(n);
    cudaSetDevice(devices.at(0));
    gpuErrchk(cudaPeekAtLastError());
    auto ptr_profile = thrust::device_ptr<float>(profile_dev[devices.at(0)]);
    auto ptr_index = thrust::device_ptr<unsigned int>(profile_idx_dev[devices.at(0)]);
    auto ptr_merged = thrust::device_ptr<unsigned long long int>(profile_A_merged[devices.at(0)]);
    auto iter_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile, ptr_index, ptr_merged));
    auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile + n, ptr_index + n, ptr_merged + n));
    for(int i = 0; i < devices.size(); ++i) {
        cudaSetDevice(devices.at(i));
        gpuErrchk(cudaPeekAtLastError());
        if (i != 0) {
            cudaMemcpy(partial_profile_host.data(), profile_merged[devices.at(i)], n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(profile_merged[devices.at(i)]);
            gpuErrchk(cudaPeekAtLastError());
            cudaSetDevice(devices.at(0));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpy(profile_merged[0], partial_profile_host.data(), n * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
            gpuErrchk(cudaPeekAtLastError());
        }
        thrust::for_each(iter_begin, iter_end, max_with_index());
        gpuErrchk(cudaPeekAtLastError());
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    cudaSetDevice(devices.at(0));
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    cudaSetDevice(devices.at(0));
    gpuErrchk(cudaPeekAtLastError());
*/
         
    // Compute the final distance calculation to convert cross correlation computed earlier into euclidean distance
    cross_correlation_to_ed<<<dim3(ceil(n / (float) WORK_SIZE), 1, 1), dim3(WORK_SIZE, 1, 1)>>>(profile_A_dev[devices.at(0)], tile_n, m); 
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(profile_idx_h.data(), profile_idx_A_dev[devices.at(0)], sizeof(unsigned int) * tile_n, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(profile_h.data(), profile_A_dev[devices.at(0)], sizeof(float) * tile_n, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_idx_A_dev[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_idx_B_dev[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_A_dev[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_B_dev[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_A_merged[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_B_merged[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());

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
        fprintf(f1, "%f\n", profile[i]);
        fprintf(f2, "%u\n", profile_idx[i] + 1);
    }
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaDeviceReset());
    fclose(f1);
    fclose(f2);
    printf("Done\n");
    return 0;
}

