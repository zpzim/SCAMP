#pragma once
#include <unordered_map>
#include <vector>
#include <list>
#include "fft_helper.h"
using std::vector;
using std::unordered_map;
using std::list;
using std::pair;

namespace SCRIMP {

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

class SCRIMP_Operation
{
private:
    unordered_map<int, double*> T_A_dev, T_B_dev, QT_dev, means_A, means_B, norms_A, norms_B, df_A, df_B, dg_A, dg_B, scratchpad;
    unordered_map<int, float*> profile_A_dev, profile_B_dev;
    unordered_map<int, unsigned long long int*> profile_A_merged, profile_B_merged;
    unordered_map<int, unsigned int*> profile_idx_A_dev, profile_idx_B_dev;
    unordered_map<int, cudaEvent_t> clocks_start, clocks_end, copy_to_host_done;
    unordered_map<int, cudaStream_t> streams;
    unordered_map<int, fft_precompute_helper*> scratch;
    size_t size_A;
    size_t size_B;
    size_t tile_size;
    size_t tile_n_x;
    size_t tile_n_y;
    size_t m;
    const bool self_join;
    const size_t MAX_TILE_SIZE;
    vector<int> devices;

    // Tile state variables
    list<pair<int,int>> tile_ordering;
    int completed_tiles;
    size_t total_tiles;
    vector<size_t> n_x;
    vector<size_t> n_y;
    vector<size_t> n_x_2;
    vector<size_t> n_y_2;
    vector<size_t> pos_x;
    vector<size_t> pos_y;
    vector<size_t> pos_x_2;
    vector<size_t> pos_y_2;

    SCRIMPError_t do_tile(SCRIMPTileType t, int device, const vector<double> &Ta_h,
                          const vector<double> &Tb_h,
                          const vector<float> &profile_h,
                          const vector<unsigned int> &profile_idx_h);

    bool pick_and_start_next_tile(int dev, const vector<double> &Ta_h,
                                  const vector<double> &Tb_h,
                                  const vector<float> &profile_h,
                                  const vector<unsigned int> &profile_idx_h);

    int issue_and_merge_tiles_on_devices(const vector<double> &Ta_host, const vector<double> &Tb_host,
                                         vector<float> &profile, vector<unsigned int> &profile_idx,
                                         vector<vector<unsigned long long int>> &profileA_h,
                                         vector<vector<unsigned long long int>> &profileB_h,
                                         int last_device_idx);

    void get_tile_ordering();

public:
    SCRIMP_Operation(size_t Asize, size_t Bsize, size_t window_sz, size_t max_tile_size, const vector<int> &dev, bool selfjoin) :
                     size_A(Asize), m(window_sz), MAX_TILE_SIZE(max_tile_size), devices(dev), self_join(selfjoin),
                     n_x(dev.size()), n_y(dev.size()), n_x_2(dev.size()), n_y_2(dev.size()),
                     pos_x(dev.size()), pos_y(dev.size()), pos_x_2(dev.size()), pos_y_2(dev.size()), completed_tiles(0)
    {
         if(self_join) {
            size_B = size_A;
         } else {
            size_B = Bsize;
         }
         tile_size = Asize / (devices.size());
         if(tile_size > MAX_TILE_SIZE) {
    	     tile_size = MAX_TILE_SIZE;
         }
         //n_y = Asize - m + 1;
         //n_x = Bsize - m + 1;
         tile_n_x = tile_size - m + 1;
         tile_n_y = tile_n_x;
    }
    SCRIMPError_t do_join(const vector<double> &Ta_host, const vector<double> &Tb_host,
                          vector<float> &profile, vector<unsigned int> &profile_idx);
    SCRIMPError_t init();
    SCRIMPError_t destroy();
};

}
