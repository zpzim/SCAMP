#pragma once
#include <unordered_map>
#include <vector>
#include "fft_helper.h"
using std::vector;
using std::unordered_map;


namespace SCRIMP {

#define MAX_TILE_SIZE 9000000
//For computing the prefix squared sum
template<class DTYPE>
struct square
{
	__host__ __device__
	DTYPE operator()(DTYPE x)
	{
		return x * x;
	}
};

template<class DTYPE, class CUFFT_DTYPE>
class SCRIMP_Operation
{
private:
    unordered_map<int, DTYPE*> T_A_dev, T_B_dev, QT_dev, means_A, means_B, stds_A, stds_B, scratchpad;
    unordered_map<int, float*> profile_A_dev, profile_B_dev;
    unordered_map<int, unsigned long long int*> profile_A_merged, profile_B_merged;
    unordered_map<int, unsigned int*> profile_idx_A_dev, profile_idx_B_dev;
    unordered_map<int, cudaEvent_t> clocks_start, clocks_end, copy_to_host_done;
    unordered_map<int, cudaStream_t> streams;
    unordered_map<int, fft_precompute_helper<DTYPE, CUFFT_DTYPE>*> scratch;
    size_t size_A;
    size_t size_B;
    size_t tile_size;
    size_t n;
    size_t tile_n;
    size_t m;
    size_t factor;
    vector<int> devices;
    SCRIMPError_t do_tile(SCRIMPTileType t, size_t t_size_x, size_t t_size_y, size_t start_x, size_t start_y, int device, const vector<DTYPE> &T_h, const vector<float> &profile_h, const vector<unsigned int> &profile_idx_h);
    bool pick_and_start_next_tile_self_join(int dev, size_t &tile_col, size_t &work_start, vector<int> &curr_tile, const size_t num_tile_rows, const size_t num_tile_cols, const vector<DTYPE> &T_h, const vector<float> &profile_h, const vector<unsigned int> &profile_idx_h, size_t &size_x, size_t &size_y, size_t &start_x, size_t &start_y);

public:
    SCRIMP_Operation(size_t Asize, size_t Bsize, size_t window_sz, const vector<int> &dev) : size_A(Asize), size_B(Bsize), m(window_sz), devices(dev) {

       tile_size = Asize / (devices.size() * 4);
       factor = ceil(tile_size / (float) MAX_TILE_SIZE);
       tile_size = tile_size / factor;
       n = Asize - m + 1;
       tile_n = tile_size - m + 1;

    }
    SCRIMPError_t do_self_join(const vector<DTYPE> &T_host, vector<float> &profile, vector<unsigned int> &profile_idx);
    SCRIMPError_t init();
    SCRIMPError_t destroy();
};

}
