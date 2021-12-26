#pragma once

#include <mutex>
#include <queue>
#include <vector>
#include "common.h"

namespace SCAMP {

struct ProfileData {
  // Only one of these should be set at once
  std::vector<uint32_t> uint32_value;
  std::vector<uint64_t> uint64_value;
  std::vector<float> float_value;
  std::vector<double> double_value;
  std::vector<std::vector<float>> matrix_value;
  std::vector<
      std::priority_queue<SCAMPmatch, std::vector<SCAMPmatch>, compareMatch>>
      match_value;
  // Unordered version of match_value
  std::vector<SCAMPmatch> match_value_unordered;
};

// Stores information about a matrix profile
class Profile {
 public:
  Profile() : type(PROFILE_TYPE_INVALID) {}
  Profile(Profile &other) {
    std::unique_lock<std::mutex> lock(_profile_lock);
    type = other.type;
    data = other.data;
  }
  Profile(Profile &&other) {
    std::unique_lock<std::mutex> lock(_profile_lock);
    type = other.type;
    data = std::move(other.data);
  }
  Profile &operator=(Profile &&other) {
    std::unique_lock<std::mutex> lock(_profile_lock);
    type = other.type;
    data = std::move(other.data);
    return *this;
  }
  Profile(SCAMPProfileType t, size_t size, float thresh_init = 0,
          int64_t mwidth = -1, int64_t mheight = -1)
      : type(t) {
    Alloc(size, mheight, mwidth, thresh_init);
  }
  std::vector<ProfileData> data;
  std::vector<float> thresholds;
  SCAMPProfileType type;
  void MergeTileToProfile(Profile *tile_profile, const OpInfo *info,
                          uint64_t position, uint64_t length,
                          uint64_t index_start, bool overflowed);
  void CopyFromDevice(const OpInfo *info, const ExecInfo *exec_info,
                      const DeviceProfile *device_tile_profile,
                      uint64_t length);
  void Alloc(size_t size, int64_t matrix_height, int64_t matrix_width,
             float default_thresh);

 private:
  void threshold_merge(const std::vector<SCAMPmatch> &matches,
                       uint64_t merge_start_col, int64_t max_matches);
  void match_merge(const std::vector<SCAMPmatch> &matches,
                   uint64_t merge_start_row, uint64_t merge_start_col,
                   int64_t max_matches);
  void matrix_merge(const std::vector<float> &values);
  std::mutex _profile_lock;
};

}  // namespace SCAMP
