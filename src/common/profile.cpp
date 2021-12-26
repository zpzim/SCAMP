#include "profile.h"

namespace SCAMP {

template <typename T>
void elementwise_sum(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    mp_full[i + merge_start] += to_merge[i];
  }
}

template <typename T>
void elementwise_max(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge, uint64_t index_offset) {
  for (int i = 0; i < tile_sz; ++i) {
    mp_entry e1, e2;
    e1.ulong = mp_full[i + merge_start];
    e2.ulong = to_merge[i];
    if (e1.floats[0] < e2.floats[0]) {
      e2.ints[1] += index_offset;
      mp_full[i + merge_start] = e2.ulong;
    }
  }
}

template <typename T>
void elementwise_max(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    if (mp_full[i + merge_start] < to_merge[i]) {
      mp_full[i + merge_start] = to_merge[i];
    }
  }
}

// Updates the adaptive thresholds of a tile in the case that there was an
// overflow. This is very similar logic to match_merge, except it does not
// add any elements to the top K lists and only updates the thresholds.
void Profile::threshold_merge(const std::vector<SCAMPmatch> &matches,
                              uint64_t merge_start_col, int64_t max_matches) {
  if (matches.empty()) {
    return;
  }

  int i = 0;
  while (i < matches.size()) {
    uint64_t curr_col = matches[i].col;
    int count = 1;
    // Count how many results we have for the current column.
    while (i + count < matches.size() && matches[i + count].col == curr_col) {
      ++count;
    }
    // If we have more than max_matches we can update the threshold with the
    // smallest value.
    if (count - 1 > max_matches && thresholds[curr_col + merge_start_col] <
                                       matches[i + max_matches].corr) {
      thresholds[curr_col + merge_start_col] = matches[i + max_matches].corr;
    }
    i += count;
  }
}

// Merges the elements in matches into the top K values in the current
// profile. Matches must be properly sorted first by column in ascending
// order, then by correlation in descending order.
void Profile::match_merge(const std::vector<SCAMPmatch> &matches,
                          uint64_t merge_start_row, uint64_t merge_start_col,
                          int64_t max_matches) {
  if (matches.empty()) {
    return;
  }

  int i = 0;
  while (i < matches.size()) {
    uint64_t curr_col = matches[i].col;
    auto &pq = this->data.front().match_value[curr_col + merge_start_col];
    uint64_t count = 0;
    float old_val;
    bool update_possible = false;
    // Loop over the initial values that might need to go into the top K
    while (i + count < matches.size() && matches[i + count].col == curr_col &&
           count < max_matches) {
      auto &match = matches[i + count];
      // If the match is not better than the bottom of the top K, break out.
      if (pq.size() == max_matches && match.corr <= pq.top().corr) {
        break;
      }
      // If we have found K values we need to make some space for the new one.
      if (pq.size() == max_matches) {
        update_possible = true;
        old_val = pq.top().corr;
        pq.pop();
      }
      pq.emplace(match.corr, match.row + merge_start_row,
                 match.col + merge_start_col);
      ++count;
    }

    // Skip the rest of the values for this column, they aren't useful.
    while (i + count < matches.size() && matches[i + count].col == curr_col) {
      ++count;
    }
    // If we ever updated the top K we can update the threshold.
    if (update_possible) {
      this->thresholds[curr_col + merge_start_col] = old_val;
    }
    i += count;
  }
}

// Merges elements in matches into a reduced distance matrix summary.
void Profile::matrix_merge(const std::vector<float> &values) {
  for (int i = 0; i < values.size(); ++i) {
    if (this->data[0].float_value[i] < values[i]) {
      this->data[0].float_value[i] = values[i];
    }
  }
}

void Profile::Alloc(size_t size, int64_t matrix_height, int64_t matrix_width,
                    float default_thresh) {
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      data.emplace_back();
      data[0].double_value.resize(size, 0);
      break;
    case PROFILE_TYPE_1NN:
      data.emplace_back();
      data[0].float_value.resize(size, -2.0);
      break;
    case PROFILE_TYPE_1NN_INDEX:
      mp_entry e;
      e.ints[1] = -1u;
      e.floats[0] = -2.0;
      data.emplace_back();
      data[0].uint64_value.resize(size, e.ulong);
      break;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      data.emplace_back();
      data[0].uint64_value.resize(size, 0);
      break;
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      data.emplace_back();
      data[0].match_value.resize(size);
      thresholds.resize(size, default_thresh);
      break;
    case PROFILE_TYPE_MATRIX_SUMMARY:
      data.emplace_back();
      data[0].float_value.resize(matrix_height * matrix_width, -2.0);
      break;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      break;
  }
}

// Copies a profile to the host
void Profile::CopyFromDevice(const OpInfo *info, const ExecInfo *exec_info,
                             const DeviceProfile *device_tile_profile,
                             uint64_t length) {
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      Memcopy(this->data[0].double_value.data(),
              device_tile_profile->at(PROFILE_TYPE_SUM_THRESH),
              length * sizeof(double), true, exec_info);
      break;
    case PROFILE_TYPE_1NN:
      Memcopy(this->data[0].float_value.data(),
              device_tile_profile->at(PROFILE_TYPE_1NN), length * sizeof(float),
              true, exec_info);
      break;
    case PROFILE_TYPE_1NN_INDEX:
      Memcopy(this->data[0].uint64_value.data(),
              device_tile_profile->at(PROFILE_TYPE_1NN_INDEX),
              length * sizeof(uint64_t), true, exec_info);
      break;
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      this->data[0].match_value_unordered.resize(length);
      Memcopy(this->data[0].match_value_unordered.data(),
              device_tile_profile->at(PROFILE_TYPE_APPROX_ALL_NEIGHBORS),
              length * sizeof(SCAMPmatch), true, exec_info);
      break;
    case PROFILE_TYPE_MATRIX_SUMMARY:
      Memcopy(this->data[0].float_value.data(),
              device_tile_profile->at(PROFILE_TYPE_MATRIX_SUMMARY),
              info->matrix_width * info->matrix_height * sizeof(float), true,
              exec_info);
      break;
    case PROFILE_TYPE_FREQUENCY_THRESH:
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      break;
  }
}

// Merges a profile corresponding to the result of a tile into this
// profile
void Profile::MergeTileToProfile(Profile *tile_profile, const OpInfo *info,
                                 uint64_t position, uint64_t length,
                                 uint64_t index_start, bool overflowed) {
  // Lock the before we merge, this function can be called by multiple
  // threads
  std::unique_lock<std::mutex> mlock(this->_profile_lock);

  if (type != tile_profile->type) {
    throw(SCAMPException("Profile Types do not match"));
  }
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      elementwise_sum<double>(this->data[0].double_value.data(), position,
                              length,
                              tile_profile->data[0].double_value.data());
      return;
    case PROFILE_TYPE_1NN_INDEX:
      elementwise_max<uint64_t>(
          this->data[0].uint64_value.data(), position, length,
          tile_profile->data[0].uint64_value.data(), index_start);
      return;
    case PROFILE_TYPE_1NN:
      elementwise_max<float>(this->data[0].float_value.data(), position, length,
                             tile_profile->data[0].float_value.data());
      return;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      elementwise_sum<uint64_t>(this->data[0].uint64_value.data(), position,
                                length,
                                tile_profile->data[0].uint64_value.data());
      return;
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      if (overflowed) {
        threshold_merge(tile_profile->data[0].match_value_unordered, position,
                        info->max_matches_per_column);
      } else {
        match_merge(tile_profile->data[0].match_value_unordered, index_start,
                    position, info->max_matches_per_column);
      }
      return;
    case PROFILE_TYPE_MATRIX_SUMMARY:
      matrix_merge(tile_profile->data[0].float_value);
      return;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      return;
  }
}

}  // namespace SCAMP
