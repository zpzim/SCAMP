
#include <cinttypes>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "SCAMP.h"
#include "common.h"
#include "tile.h"
#include "utils.h"
using std::vector;

namespace SCAMP {

static const int ISSUED_ALL_DEVICES = -2;

SCAMPError_t SCAMP_Operation::init() {
  for (auto device : devices) {
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());

    T_A_dev.insert({device, nullptr});
    T_B_dev.insert({device, nullptr});
    QT_dev.insert({device, nullptr});
    means_A.insert({device, nullptr});
    means_B.insert({device, nullptr});
    norms_A.insert({device, nullptr});
    norms_B.insert({device, nullptr});
    df_A.insert({device, nullptr});
    df_B.insert({device, nullptr});
    dg_A.insert({device, nullptr});
    dg_B.insert({device, nullptr});
    profile_A_dev.insert({device, nullptr});
    profile_B_dev.insert({device, nullptr});
    scratchpad.insert({device, nullptr});

    cudaMalloc(&T_A_dev.at(device), sizeof(double) * tile_size);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&T_B_dev.at(device), sizeof(double) * tile_size);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&profile_A_dev.at(device), sizeof(float) * tile_n_x);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&profile_B_dev.at(device), sizeof(float) * tile_n_y);
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
    cudaMalloc(&scratchpad.at(device), sizeof(double) * tile_size);
    scratch[device] =
        std::make_shared<fft_precompute_helper>(tile_size, m, true);
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
    cudaFree(scratchpad.at(device));
    cudaEventDestroy(clocks_start[device]);
    cudaEventDestroy(clocks_end[device]);
    cudaEventDestroy(copy_to_host_done[device]);
    cudaStreamDestroy(streams.at(device));
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Operation::do_tile(SCAMPTileType t, int device,
                                      const vector<double> &Ta_h,
                                      const vector<double> &Tb_h) {
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
  cudaMemsetAsync(profile_A_dev[device], 0, sizeof(uint32_t) * t_n_x,
                  streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemsetAsync(profile_B_dev[device], 0, sizeof(uint32_t) * t_n_y,
                  streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
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
  SCAMP_Tile tile(t, T_A_dev[device], T_B_dev[device], df_A[device],
                  df_B[device], dg_A[device], dg_B[device], norms_A[device],
                  norms_B[device], means_A[device], means_B[device],
                  QT_dev[device], profile_A_dev[device], profile_B_dev[device],
                  start_x, start_y, tile_start_col_position,
                  tile_start_row_position, n_y[device], n_x[device], m,
                  scratch[device], dev_props.at(device), fp_type, thresh);
  cudaEventRecord(clocks_start[device], streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  err = tile.execute(streams.at(device));
  cudaEventRecord(clocks_end[device], streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  return err;
}

void SCAMP_Operation::get_tile_ordering() {
  tile_ordering.clear();
  size_t num_tile_rows = ceil((size_B - m + 1) / static_cast<double>(tile_n_y));
  size_t num_tile_cols = ceil((size_A - m + 1) / static_cast<double>(tile_n_x));

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

bool SCAMP_Operation::pick_and_start_next_tile(int dev,
                                               const vector<double> &Ta_h,
                                               const vector<double> &Tb_h) {
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
      err = do_tile(SELF_JOIN_UPPER_TRIANGULAR, dev, Ta_h, Tb_h);
    } else {
      // full tile
      err = do_tile(SELF_JOIN_FULL_TILE, dev, Ta_h, Tb_h);
    }
  } else if (full_join) {
    // BiDirectional AB-join
    err = do_tile(AB_FULL_JOIN_FULL_TILE, dev, Ta_h, Tb_h);
  } else {
    // Column AB-join
    err = do_tile(AB_JOIN_FULL_TILE, dev, Ta_h, Tb_h);
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
    vector<uint32_t> &profile_A_full_host,
    vector<uint32_t> &profile_B_full_host, vector<vector<uint32_t>> &profileA_h,
    vector<vector<uint32_t>> &profileB_h,
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
    cudaMemcpyAsync(profileA_h.at(i).data(), profile_A_dev[device],
                    sizeof(uint32_t) * (n_x[device] - m + 1),
                    cudaMemcpyDeviceToHost, streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
    if (self_join || full_join) {
      cudaMemcpyAsync(profileB_h.at(i).data(), profile_B_dev[device],
                      sizeof(uint32_t) * (n_y[device] - m + 1),
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
      done = pick_and_start_next_tile(device, Ta_host, Tb_host);
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
    elementwise_sum(profile_A_full_host, pos_x_2[device], n_x_2[device] - m + 1,
                    &profileA_h.at(i));
    gpuErrchk(cudaPeekAtLastError());
    if (self_join) {
      elementwise_sum(profile_A_full_host, pos_y_2[device],
                      n_y_2[device] - m + 1, &profileB_h.at(i));
      gpuErrchk(cudaPeekAtLastError());
    } else if (full_join) {
      elementwise_sum(profile_B_full_host, pos_y_2[device],
                      n_y_2[device] - m + 1, &profileB_h.at(i));
      gpuErrchk(cudaPeekAtLastError());
    }
    completed_tiles++;
    printf("%f percent complete\n",
           (completed_tiles / static_cast<float>(total_tiles)) * 100);
  }
  return last_dev;
}

SCAMPError_t SCAMP_Operation::do_join(const vector<double> &Ta_host,
                                      const vector<double> &Tb_host,
                                      vector<uint32_t> &profile,
                                      vector<uint32_t> &profile_B) {
  vector<vector<uint32_t>> profileA_h(devices.size(),
                                      vector<uint32_t>(tile_n_y)),
      profileB_h(devices.size(), vector<uint32_t>(tile_n_x));
  bool done = false;
  int last_dev = ISSUED_ALL_DEVICES;
  get_tile_ordering();
  printf("Performing join with %lu tiles.\n", tile_ordering.size());
  for (int i = 0; i < devices.size(); ++i) {
    int device = devices.at(i);
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());
    done = pick_and_start_next_tile(device, Ta_host, Tb_host);
    gpuErrchk(cudaPeekAtLastError());
    if (done) {
      last_dev = i;
      break;
    }
  }

  while (last_dev == ISSUED_ALL_DEVICES) {
    last_dev = issue_and_merge_tiles_on_devices(
        Ta_host, Tb_host, profile, profile_B, profileA_h, profileB_h);
  }

  issue_and_merge_tiles_on_devices(Ta_host, Tb_host, profile, profile_B,
                                   profileA_h, profileB_h, last_dev);

  return SCAMP_NO_ERROR;
}

void do_SCAMP(const vector<double> &Ta_h, const vector<double> &Tb_h,
              vector<uint32_t> *profile_h, vector<uint32_t> *profile_B_h,
              const unsigned int m, const size_t max_tile_size,
              const vector<int> &devices, bool self_join, FPtype t,
              bool full_join, size_t start_row, size_t start_col,
              float thresh) {
  if (devices.empty()) {
    printf("Error: no gpu provided\n");
    exit(0);
  }
  // Allocate and initialize memory
  clock_t start, end;
  SCAMP_Operation op(Ta_h.size(), Tb_h.size(), m, max_tile_size, devices,
                     self_join, t, full_join, start_row, start_col, thresh);
  op.init();
  gpuErrchk(cudaPeekAtLastError());
  start = clock();
  if (self_join) {
    op.do_join(Ta_h, Ta_h, *profile_h, *profile_B_h);
  } else {
    op.do_join(Ta_h, Tb_h, *profile_h, *profile_B_h);
  }
  cudaDeviceSynchronize();
  end = clock();
  gpuErrchk(cudaPeekAtLastError());
  op.destroy();
  gpuErrchk(cudaPeekAtLastError());
  printf(
      "Finished SCAMP to generate partial matrix profile of size %lu in %f "
      "seconds on %lu devices:\n",
      profile_h->size(), (end - start) / static_cast<double>(CLOCKS_PER_SEC),
      devices.size());
}

}  // namespace SCAMP
