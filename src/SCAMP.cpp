#include "SCAMP.h"
#include <float.h>
#include <numeric>
#include <unordered_map>
#include <vector>
#include "common.h"
#include "tile.h"
#include "utils.h"
using std::make_pair;
using std::unordered_map;
using std::vector;

namespace SCAMP {

static const int ISSUED_ALL_DEVICES = -2;

SCAMPError_t SCAMP_Operation::init() {
  for (auto device : devices) {
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());

    T_A_dev.insert(make_pair(device, (double *)0));
    T_B_dev.insert(make_pair(device, (double *)0));
    QT_dev.insert(make_pair(device, (double *)0));
    means_A.insert(make_pair(device, (double *)0));
    means_B.insert(make_pair(device, (double *)0));
    norms_A.insert(make_pair(device, (double *)0));
    norms_B.insert(make_pair(device, (double *)0));
    df_A.insert(make_pair(device, (double *)0));
    df_B.insert(make_pair(device, (double *)0));
    dg_A.insert(make_pair(device, (double *)0));
    dg_B.insert(make_pair(device, (double *)0));
    profile_A_dev.insert(make_pair(device, (float *)NULL));
    profile_B_dev.insert(make_pair(device, (float *)NULL));
    profile_A_merged.insert(make_pair(device, (unsigned long long int *)NULL));
    profile_B_merged.insert(make_pair(device, (unsigned long long int *)NULL));
    profile_idx_A_dev.insert(make_pair(device, (unsigned int *)NULL));
    profile_idx_B_dev.insert(make_pair(device, (unsigned int *)NULL));
    scratchpad.insert(make_pair(device, (double *)NULL));

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
    cudaMalloc(&profile_A_merged.at(device),
               sizeof(unsigned long long int) * tile_n_x);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&profile_B_merged.at(device),
               sizeof(unsigned long long int) * tile_n_y);
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

SCAMPError_t SCAMP_Operation::destroy() {
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

SCAMPError_t SCAMP_Operation::do_tile(
    SCAMPTileType t, int device, const vector<double> &Ta_h,
    const vector<double> &Tb_h, const vector<float> &profile_h,
    const vector<unsigned int> &profile_idx_h, const vector<float> &profile_B_h,
    const vector<unsigned int> &profile_idx_B_h) {
  size_t start_x = pos_x[device];
  size_t start_y = pos_y[device];
  SCAMPError_t err;
  size_t t_n_x = n_x[device] - m + 1;
  size_t t_n_y = n_y[device] - m + 1;
  printf("tile type = %d start_pos = [%lu, %lu]...\n", t, start_y, start_x);
  cudaMemcpyAsync(T_A_dev[device], Ta_h.data() + start_x,
                  sizeof(double) * n_x[device], cudaMemcpyHostToDevice,
                  streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(T_B_dev[device], Tb_h.data() + start_y,
                  sizeof(double) * n_y[device], cudaMemcpyHostToDevice,
                  streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(profile_A_dev[device], profile_h.data() + start_x,
                  sizeof(float) * t_n_x, cudaMemcpyHostToDevice,
                  streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(profile_idx_A_dev[device], profile_idx_h.data() + start_x,
                  sizeof(unsigned int) * t_n_x, cudaMemcpyHostToDevice,
                  streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  if (self_join) {
    cudaMemcpyAsync(profile_B_dev[device], profile_h.data() + start_y,
                    sizeof(float) * t_n_y, cudaMemcpyHostToDevice,
                    streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpyAsync(profile_idx_B_dev[device], profile_idx_h.data() + start_y,
                    sizeof(unsigned int) * t_n_y, cudaMemcpyHostToDevice,
                    streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
  } else if (full_join) {
    cudaMemcpyAsync(profile_B_dev[device], profile_B_h.data() + start_y,
                    sizeof(float) * t_n_y, cudaMemcpyHostToDevice,
                    streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
    printf("start = %lu, size = %lu\n", start_y, profile_idx_B_h.size());
    cudaMemcpyAsync(profile_idx_B_dev[device], profile_idx_B_h.data() + start_y,
                    sizeof(unsigned int) * t_n_y, cudaMemcpyHostToDevice,
                    streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
  }
  // FIXME?: Computing the sliding dot products & statistics for each tile is
  // overkill
  compute_statistics(T_A_dev[device], norms_A[device], df_A[device],
                     dg_A[device], means_A[device], t_n_x, m,
                     streams.at(device), scratchpad[device]);
  gpuErrchk(cudaPeekAtLastError());
  compute_statistics(T_B_dev[device], norms_B[device], df_B[device],
                     dg_B[device], means_B[device], t_n_y, m,
                     streams.at(device), scratchpad[device]);
  gpuErrchk(cudaPeekAtLastError());
  launch_merge_mp_idx(profile_A_dev[device], profile_idx_A_dev[device], t_n_x,
                      profile_A_merged[device], streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  if (self_join || full_join) {
    launch_merge_mp_idx(profile_B_dev[device], profile_idx_B_dev[device], t_n_y,
                        profile_B_merged[device], streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
  }
  SCAMP_Tile tile(
      t, T_A_dev[device], T_B_dev[device], df_A[device], df_B[device],
      dg_A[device], dg_B[device], norms_A[device], norms_B[device],
      means_A[device], means_B[device], QT_dev[device],
      profile_A_merged[device], profile_B_merged[device], start_x, start_y,
      tile_start_col_position, tile_start_row_position, n_y[device],
      n_x[device], m, scratch[device], dev_props.at(device), fp_type);
  cudaEventRecord(clocks_start[device], streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  err = tile.execute(streams.at(device));
  cudaEventRecord(clocks_end[device], streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  return err;
}

void SCAMP_Operation::get_tile_ordering() {
  tile_ordering.clear();
  size_t num_tile_rows = ceil((size_B - m + 1) / (float)tile_n_y);
  size_t num_tile_cols = ceil((size_A - m + 1) / (float)tile_n_x);

  if (self_join) {
    for (int offset = 0; offset < num_tile_rows - 1; ++offset) {
      for (int diag = 0; diag < num_tile_cols - 1 - offset; ++diag) {
        tile_ordering.emplace_back(diag, diag + offset);
      }
    }

    for (int i = 0; i < num_tile_rows; ++i) {
      tile_ordering.emplace_back(i, num_tile_cols - 1);
    }
  } else {
    // Add upper diagonals one at a time except for edge tiles
    for (int diag = 0; diag < num_tile_cols - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_cols - 1 && offset < num_tile_rows - 1;
           ++offset) {
        tile_ordering.emplace_back(offset, diag + offset);
      }
    }

    // Add lower diagonals one at a time except for edge tiles
    for (int diag = 1; diag < num_tile_rows - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_rows - 1 && offset < num_tile_cols - 1;
           ++offset) {
        tile_ordering.emplace_back(offset + diag, offset);
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
    while (x < num_tile_cols - 1) {
      tile_ordering.emplace_back(num_tile_rows - 1, x);
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

bool SCAMP_Operation::pick_and_start_next_tile(
    int dev, const vector<double> &Ta_h, const vector<double> &Tb_h,
    const vector<float> &profile_h, const vector<unsigned int> &profile_idx_h,
    const vector<float> &profile_B_h,
    const vector<unsigned int> &profile_idx_B_h) {
  bool done = false;
  int tile_row = tile_ordering.front().first;
  int tile_col = tile_ordering.front().second;
  // Get the position of the tile we will compute
  pos_x[dev] = tile_col * tile_n_x;
  pos_y[dev] = tile_row * tile_n_y;
  // Get the size of the tile we will compute
  n_x[dev] = std::min(tile_size, size_A - pos_x[dev]);
  n_y[dev] = std::min(tile_size, size_B - pos_y[dev]);
  SCAMPError_t err;
  if (self_join) {
    if (tile_row == tile_col) {
      // partial tile on diagonal
      err = do_tile(SELF_JOIN_UPPER_TRIANGULAR, dev, Ta_h, Tb_h, profile_h,
                    profile_idx_h, profile_B_h, profile_idx_B_h);
    } else {
      // full tile
      err = do_tile(SELF_JOIN_FULL_TILE, dev, Ta_h, Tb_h, profile_h,
                    profile_idx_h, profile_B_h, profile_idx_B_h);
    }
  } else if (full_join) {
    // BiDirectional AB-join
    err = do_tile(AB_FULL_JOIN_FULL_TILE, dev, Ta_h, Tb_h, profile_h,
                  profile_idx_h, profile_B_h, profile_idx_B_h);
  } else {
    // Column AB-join
    err = do_tile(AB_JOIN_FULL_TILE, dev, Ta_h, Tb_h, profile_h, profile_idx_h,
                  profile_B_h, profile_idx_B_h);
  }
  if (err != SCAMP_NO_ERROR) {
    printf("ERROR %d executing tile. \n", err);
  }
  tile_ordering.pop_front();
  if (tile_ordering.empty()) {
    done = true;
  }
  return done;
}

int SCAMP_Operation::issue_and_merge_tiles_on_devices(
    const vector<double> &Ta_host, const vector<double> &Tb_host,
    vector<float> &profile_A_full_host,
    vector<unsigned int> &profile_idx_A_full_host,
    vector<float> &profile_B_full_host,
    vector<unsigned int> &profile_idx_B_full_host,
    vector<vector<unsigned long long int>> &profileA_h,
    vector<vector<unsigned long long int>> &profileB_h,
    int last_device_idx = ISSUED_ALL_DEVICES) {
  bool done = last_device_idx != ISSUED_ALL_DEVICES;
  int last_dev = ISSUED_ALL_DEVICES;
  if (last_device_idx == ISSUED_ALL_DEVICES) {
    last_device_idx = devices.size() - 1;
  }
  for (int i = 0; i <= last_device_idx; ++i) {
    int device = devices.at(i);
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpyAsync(profileA_h.at(i).data(), profile_A_merged[device],
                    sizeof(unsigned long long int) * (n_x[device] - m + 1),
                    cudaMemcpyDeviceToHost, streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
    if (self_join || full_join) {
      cudaMemcpyAsync(profileB_h.at(i).data(), profile_B_merged[device],
                      sizeof(unsigned long long int) * (n_y[device] - m + 1),
                      cudaMemcpyDeviceToHost, streams.at(device));
      gpuErrchk(cudaPeekAtLastError());
    }
    cudaEventRecord(copy_to_host_done[device], streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
    n_x_2[device] = n_x[device];
    n_y_2[device] = n_y[device];
    pos_x_2[device] = pos_x[device];
    pos_y_2[device] = pos_y[device];
    if (!done) {
      done = pick_and_start_next_tile(
          device, Ta_host, Tb_host, profile_A_full_host,
          profile_idx_A_full_host, profile_B_full_host,
          profile_idx_B_full_host);
      if (done) {
        last_dev = i;
      }
    }
  }

  for (int i = 0; i <= last_device_idx; ++i) {
    int device = devices.at(i);
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());
    cudaEventSynchronize(copy_to_host_done[device]);
    gpuErrchk(cudaPeekAtLastError());
    printf("%lu %lu, %lu, %lu, %lu, %lu\n", profile_idx_A_full_host.size(),
           profileA_h.at(i).size(), pos_x_2[device], n_x_2[device] - m + 1,
           pos_y_2[device], n_y_2[device] - m + 1);
    elementwise_max_with_index(profile_A_full_host, profile_idx_A_full_host,
                               pos_x_2[device], n_x_2[device] - m + 1,
                               &profileA_h.at(i));
    gpuErrchk(cudaPeekAtLastError());
    if (self_join) {
      printf("Second Elementwise\n");
      elementwise_max_with_index(profile_A_full_host, profile_idx_A_full_host,
                                 pos_y_2[device], n_y_2[device] - m + 1,
                                 &profileB_h.at(i));
      gpuErrchk(cudaPeekAtLastError());
    } else if (full_join) {
      elementwise_max_with_index(profile_B_full_host, profile_idx_B_full_host,
                                 pos_y_2[device], n_y_2[device] - m + 1,
                                 &profileB_h.at(i));
      gpuErrchk(cudaPeekAtLastError());
    }
    completed_tiles++;
    printf("%f percent complete\n",
           (completed_tiles / (float)total_tiles) * 100);
  }
  return last_dev;
}

SCAMPError_t SCAMP_Operation::do_join(const vector<double> &Ta_host,
                                      const vector<double> &Tb_host,
                                      vector<float> &profile,
                                      vector<unsigned int> &profile_idx,
                                      vector<float> &profile_B,
                                      vector<unsigned int> &profile_idx_B) {
  vector<vector<unsigned long long int>> profileA_h(
      devices.size(), vector<unsigned long long int>(tile_n_y)),
      profileB_h(devices.size(), vector<unsigned long long int>(tile_n_x));
  bool done = false;
  int last_dev = ISSUED_ALL_DEVICES;
  get_tile_ordering();
  printf("Performing join with %lu tiles.\n", tile_ordering.size());
  for (int i = 0; i < devices.size(); ++i) {
    int device = devices.at(i);
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());
    done = pick_and_start_next_tile(device, Ta_host, Tb_host, profile,
                                    profile_idx, profile_B, profile_idx_B);
    gpuErrchk(cudaPeekAtLastError());
    if (done) {
      last_dev = i;
      break;
    }
  }

  while (last_dev == ISSUED_ALL_DEVICES) {
    last_dev = issue_and_merge_tiles_on_devices(
        Ta_host, Tb_host, profile, profile_idx, profile_B, profile_idx_B,
        profileA_h, profileB_h);
  }

  issue_and_merge_tiles_on_devices(Ta_host, Tb_host, profile, profile_idx,
                                   profile_B, profile_idx_B, profileA_h,
                                   profileB_h, last_dev);

  return SCAMP_NO_ERROR;
}

void do_SCAMP(const vector<double> &Ta_h, const vector<double> &Tb_h,
              vector<float> &profile_h, vector<unsigned int> &profile_idx_h,
              vector<float> &profile_B_h, vector<unsigned int> &profile_idx_B_h,
              const unsigned int m, const size_t max_tile_size,
              const vector<int> &devices, bool self_join, FPtype t,
              bool full_join, size_t start_row, size_t start_col) {
  if (devices.empty()) {
    printf("Error: no gpu provided\n");
    exit(0);
  }
  // Allocate and initialize memory
  clock_t start, end;
  SCAMP_Operation op(Ta_h.size(), Tb_h.size(), m, max_tile_size, devices,
                     self_join, t, full_join, start_row, start_col);
  op.init();
  gpuErrchk(cudaPeekAtLastError());
  start = clock();
  if (self_join) {
    op.do_join(Ta_h, Ta_h, profile_h, profile_idx_h, profile_B_h,
               profile_idx_B_h);
  } else {
    op.do_join(Ta_h, Tb_h, profile_h, profile_idx_h, profile_B_h,
               profile_idx_B_h);
  }
  cudaDeviceSynchronize();
  end = clock();
  gpuErrchk(cudaPeekAtLastError());
  op.destroy();
  gpuErrchk(cudaPeekAtLastError());
  printf(
      "Finished SCAMP to generate partial matrix profile of size %lu in %f "
      "seconds on %lu devices:\n",
      profile_h.size(), (end - start) / (double)CLOCKS_PER_SEC, devices.size());
}

}  // namespace SCAMP
