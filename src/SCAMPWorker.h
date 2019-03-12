#pragma once
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "common.h"
#include "SCAMP.pb.h"
#include "fft_helper.h"

namespace SCAMP {
 
class Worker {
  private:
    SCAMPArchitecture _arch;
    int _id;
    int _cuda_id;
    void init_cuda();
    void free_cuda();
    void init_cpu();
    void free_cpu();
    Profile AllocProfile(SCAMPProfileType t, uint64_t size);
    void CopyProfileToHost(Profile *destination_profile, const DeviceProfile *device_tile_profile, uint64_t length);
    void MergeTileIntoFullProfile(Profile *tile_profile,
                                               uint64_t position,
                                               uint64_t length,
                                               Profile *full_profile,
                                               uint64_t index_start, std::mutex &lock);
  public:
    Worker(int max_tile_width, int max_tile_height, int tile_ts_length, int mp_window, SCAMPProfileType profile_type, int id, SCAMPArchitecture arch, int cuda_id = -1) :
           _max_tile_width(max_tile_width), _max_tile_height(max_tile_height),
           _max_tile_ts_size(tile_ts_length), _arch(arch), _cuda_id(cuda_id), 
           _mp_window(mp_window), _profile_type(profile_type),
           _current_tile_width(0), _current_tile_height(0), _current_tile_row(0), _current_tile_col(0) {

      _profile_a_tile_dev[_profile_type] = nullptr;
      _profile_b_tile_dev[_profile_type] = nullptr;
      _profile_a_tile = AllocProfile(_profile_type, _max_tile_height);
      _profile_b_tile = AllocProfile(_profile_type, _max_tile_width);

      switch(_arch) {
      case CUDA_GPU_WORKER:
        init_cuda();
        break;
      case CPU_WORKER:
        init_cpu();
        break;
     } 
    }
    ~Worker() {
      switch(_arch) {
      case CUDA_GPU_WORKER:
        free_cuda();
        break;
      case CPU_WORKER:
        free_cpu();
        break;
      }
    }
    size_t get_tile_width() { return _current_tile_width; }
    size_t get_tile_height() { return _current_tile_height; }
    size_t get_tile_row() { return _current_tile_row; }
    size_t get_tile_col() { return _current_tile_col; }
    void set_tile_col(size_t col) { _current_tile_col = col; }
    void set_tile_row(size_t row) { _current_tile_row = row; }
    void set_tile_height(size_t height) { _current_tile_height = height; }
    void set_tile_width(size_t width) { _current_tile_width = width; }
    void MergeProfile(bool self_join, bool fetch_rows, bool keep_rows, Profile *profile_a, std::mutex &a_lock,  Profile *profile_b, std::mutex &b_lock);
    void Init();
    void Sync();
    void InitStats(const PrecomputedInfo& a, const PrecomputedInfo& b);
    SCAMPError_t InitProfile(Profile *profile_a, Profile *profile_b, bool self_join, bool computing_rows, bool keep_rows);
    void InitTimeseries(const google::protobuf::RepeatedField<double> &Ta_h, const google::protobuf::RepeatedField<double> &Tb_h);
    // TODO: make these private members
    double *_T_A_dev, * _T_B_dev, *_QT_dev, *_means_A, *_means_B, *_norms_A, *_norms_B, *_df_A, *_df_B, *_dg_A, *_dg_B, *_scratchpad;
    DeviceProfile _profile_a_tile_dev, _profile_b_tile_dev;
    Profile _profile_a_tile, _profile_b_tile;
    size_t _current_tile_width;
    size_t _current_tile_height;
    size_t _current_tile_col;
    size_t _current_tile_row;
    size_t _max_tile_ts_size;
    size_t _max_tile_height;
    size_t _max_tile_width;
    size_t _mp_window;
    SCAMPProfileType _profile_type;
    std::shared_ptr<fft_precompute_helper> _scratch;
#ifdef _HAS_CUDA_
    // TODO(zpzim): SCAMP_Operation should not be required to use cuda streams,
    // this is preventing us from adding a CPU codepath to SCAMP
    cudaStream_t _stream;
    // TODO(zpzim): device properties should allow CPUs to be listed as devices as
    // well.
    cudaDeviceProp _dev_props;
#endif
};
     
} // namespace SCAMP
