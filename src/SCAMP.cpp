
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

template <typename T>
void elementwise_sum(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  // printf("%lu, %lu\n", merge_start, tile_sz);
  // printf("%d\n", to_merge->size());
  for (int i = 0; i < tile_sz; ++i) {
    //    printf("%lf\n", static_cast<double>((*to_merge)[i]));
    (*mp_full)[i + merge_start] += (*to_merge)[i];
    (*to_merge)[i] = 0;
  }
  // printf("Done merge");
}

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
    profile_a_tile_dev.insert({device, nullptr});
    profile_b_tile_dev.insert({device, nullptr});
    scratchpad.insert({device, nullptr});

    cudaMalloc(&T_A_dev.at(device), sizeof(double) * tile_size);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&T_B_dev.at(device), sizeof(double) * tile_size);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&profile_a_tile_dev.at(device), sizeof(double) * tile_n_x);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&profile_b_tile_dev.at(device), sizeof(double) * tile_n_y);
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
    cudaFree(profile_a_tile_dev[device]);
    cudaFree(profile_b_tile_dev[device]);
    cudaFree(scratchpad.at(device));
    cudaEventDestroy(clocks_start[device]);
    cudaEventDestroy(clocks_end[device]);
    cudaEventDestroy(copy_to_host_done[device]);
    cudaStreamDestroy(streams.at(device));
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Operation::do_tile(
    SCAMPTileType t, int device,
    const google::protobuf::RepeatedField<double> &Ta_h,
    const google::protobuf::RepeatedField<double> &Tb_h) {
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
  cudaMemsetAsync(profile_a_tile_dev[device], 0, sizeof(double) * t_n_x,
                  streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemsetAsync(profile_b_tile_dev[device], 0, sizeof(double) * t_n_y,
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
  /*
    cudaDeviceSynchronize();
    std::vector<double> test(t_n_x, 0);
    cudaMemcpy(test.data(), T_B_dev[device], sizeof(double) * t_n_x,
    cudaMemcpyDeviceToHost); gpuErrchk(cudaPeekAtLastError());

    for ( int i = 0; i < t_n_x; ++i ) {
      printf("%lf\n", test[i]);
    }
    */
  printf("Threshold = %lf\n", thresh);
  SCAMP_Tile tile(
      t, T_A_dev[device], T_B_dev[device], df_A[device], df_B[device],
      dg_A[device], dg_B[device], norms_A[device], norms_B[device],
      means_A[device], means_B[device], QT_dev[device],
      profile_a_tile_dev[device], profile_b_tile_dev[device], start_x, start_y,
      tile_start_col_position, tile_start_row_position, n_y[device],
      n_x[device], m, scratch[device], dev_props.at(device), fp_type, thresh);
  cudaEventRecord(clocks_start[device], streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  err = tile.execute(streams.at(device));
  cudaEventRecord(clocks_end[device], streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  /*
    cudaDeviceSynchronize();

    std::vector<double> test(t_n_x, 0);
    cudaMemcpy(test.data(), profile_a_tile_dev[device], sizeof(double) * t_n_x,
    cudaMemcpyDeviceToHost); gpuErrchk(cudaPeekAtLastError()); for ( int i = 0;
    i < t_n_x; ++i ) { printf("%lf\n", test[i]);
    }
  */
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

bool SCAMP_Operation::pick_and_start_next_tile(
    int dev, const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b) {
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
      err =
          do_tile(SELF_JOIN_UPPER_TRIANGULAR, dev, timeseries_a, timeseries_b);
    } else {
      // full tile
      err = do_tile(SELF_JOIN_FULL_TILE, dev, timeseries_a, timeseries_b);
    }
  } else if (full_join) {
    // BiDirectional AB-join
    err = do_tile(AB_FULL_JOIN_FULL_TILE, dev, timeseries_a, timeseries_b);
  } else {
    // Column AB-join
    err = do_tile(AB_JOIN_FULL_TILE, dev, timeseries_a, timeseries_b);
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

// TODO(zpzim): make generic on device
void SCAMP_Operation::CopyProfileToHost(Profile *destination_profile,
                                        const double *device_tile_profile,
                                        uint64_t length, cudaStream_t s) {
  switch (destination_profile->type()) {
    case PROFILE_TYPE_SUM_THRESH:
      printf("Destination size = %d\n", destination_profile->mutable_data()
                                            ->Mutable(0)
                                            ->mutable_double_value()
                                            ->mutable_value()
                                            ->size());
      cudaMemcpyAsync(destination_profile->mutable_data()
                          ->Mutable(0)
                          ->mutable_double_value()
                          ->mutable_value()
                          ->mutable_data(),
                      device_tile_profile, length * sizeof(double),
                      cudaMemcpyDeviceToHost, s);
      gpuErrchk(cudaPeekAtLastError());
      return;
    case PROFILE_TYPE_1NN_INDEX:
      return;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      return;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      return;
  }
}

void SCAMP_Operation::MergeTileIntoFullProfile(Profile *tile_profile,
                                               uint64_t position,
                                               uint64_t length,
                                               Profile *full_profile) {
  switch (full_profile->type()) {
    case PROFILE_TYPE_SUM_THRESH:
      elementwise_sum<google::protobuf::RepeatedField<double>>(
          full_profile->mutable_data()
              ->Mutable(0)
              ->mutable_double_value()
              ->mutable_value(),
          position, length,
          tile_profile->mutable_data()
              ->Mutable(0)
              ->mutable_double_value()
              ->mutable_value());
      return;
    case PROFILE_TYPE_1NN_INDEX:
      // elementwise_max_with_index();
      return;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      elementwise_sum<google::protobuf::RepeatedField<uint64_t>>(
          full_profile->mutable_data()
              ->Mutable(0)
              ->mutable_uint64_value()
              ->mutable_value(),
          position, length,
          tile_profile->mutable_data()
              ->Mutable(0)
              ->mutable_uint64_value()
              ->mutable_value());
      return;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      return;
  }
}

int SCAMP_Operation::issue_and_merge_tiles_on_devices(
    const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b,
    Profile *profile_a_full, Profile *profile_b_full,
    vector<Profile> *profile_a_tile, vector<Profile> *profile_b_tile,
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
    CopyProfileToHost(&profile_a_tile->at(i), profile_a_tile_dev[device],
                      n_x[device] - m + 1, streams[device]);
    /*
        cudaMemcpyAsync(profileA_h.at(i).data(), profile_A_dev[device],
                        sizeof(double) * (n_x[device] - m + 1),
                        cudaMemcpyDeviceToHost, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
    */
    if (self_join || full_join) {
      CopyProfileToHost(&profile_b_tile->at(i), profile_b_tile_dev[device],
                        n_y[device] - m + 1, streams[device]);
      /*
            cudaMemcpyAsync(profileB_h.at(i).data(), profile_B_dev[device],
                            sizeof(double) * (n_y[device] - m + 1),
                            cudaMemcpyDeviceToHost, streams.at(device));
            gpuErrchk(cudaPeekAtLastError());
      */
    }
    cudaEventRecord(copy_to_host_done[device], streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
    n_x_2[device] = n_x[device];
    n_y_2[device] = n_y[device];
    pos_x_2[device] = pos_x[device];
    pos_y_2[device] = pos_y[device];
    if (!done) {
      done = pick_and_start_next_tile(device, timeseries_a, timeseries_b);
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
    MergeTileIntoFullProfile(&profile_a_tile->at(i), pos_x_2[device],
                             n_x_2[device] - m + 1, profile_a_full);
    /*
     elementwise_sum(profile_A_full_host, pos_x_2[device], n_x_2[device] - m +
     1, &profileA_h.at(i)); gpuErrchk(cudaPeekAtLastError());
     */
    if (self_join) {
      // printf("Hello\n");
      // printf("%d", profile_b_tile->at(i).data().size());
      MergeTileIntoFullProfile(&profile_b_tile->at(i), pos_y_2[device],
                               n_y_2[device] - m + 1, profile_a_full);
      // printf("Goodbye\n");
      /*
            elementwise_sum(profile_A_full_host, pos_y_2[device],
                            n_y_2[device] - m + 1, &profileB_h.at(i));
            //gpuErrchk(cudaPeekAtLastError());
      */
    } else if (full_join) {
      MergeTileIntoFullProfile(&profile_b_tile->at(i), pos_y_2[device],
                               n_y_2[device] - m + 1, profile_b_full);
      /*
            elementwise_sum(profile_B_full_host, pos_y_2[device],
                            n_y_2[device] - m + 1, &profileB_h.at(i));
            gpuErrchk(cudaPeekAtLastError());
      */
    }
    completed_tiles++;
    printf("%f percent complete\n",
           (completed_tiles / static_cast<float>(total_tiles)) * 100);
  }
  return last_dev;
}

Profile SCAMP_Operation::InitProfile(SCAMPProfileType t, uint64_t size) {
  Profile p;
  p.set_type(t);
  switch (t) {
    case PROFILE_TYPE_SUM_THRESH:
      p.mutable_data()->Add()->mutable_double_value()->mutable_value()->Resize(
          size, 0);
      return p;
    case PROFILE_TYPE_1NN_INDEX:
      p.mutable_data()->Add()->mutable_uint64_value()->mutable_value()->Resize(
          size, 0);
      return p;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      p.mutable_data()->Add()->mutable_uint64_value()->mutable_value()->Resize(
          size, 0);
      return p;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      return p;
  }
}

SCAMPError_t SCAMP_Operation::do_join(
    const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b,
    Profile *profile_a, Profile *profile_b) {
  printf("a type = %d\n", profile_a->type());
  printf("b type = %d\n", profile_b->type());
  vector<Profile> profile_a_tile(devices.size(),
                                 InitProfile(profile_a->type(), tile_n_y));
  vector<Profile> profile_b_tile(devices.size(),
                                 InitProfile(profile_b->type(), tile_n_x));

  bool done = false;
  int last_dev = ISSUED_ALL_DEVICES;
  get_tile_ordering();
  printf("Performing join with %lu tiles.\n", tile_ordering.size());
  for (int i = 0; i < devices.size(); ++i) {
    int device = devices.at(i);
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());
    done = pick_and_start_next_tile(device, timeseries_a, timeseries_b);
    gpuErrchk(cudaPeekAtLastError());
    if (done) {
      last_dev = i;
      break;
    }
  }

  while (last_dev == ISSUED_ALL_DEVICES) {
    last_dev = issue_and_merge_tiles_on_devices(
        timeseries_a, timeseries_b, profile_a, profile_b, &profile_a_tile,
        &profile_b_tile);
  }

  issue_and_merge_tiles_on_devices(timeseries_a, timeseries_b, profile_a,
                                   profile_b, &profile_a_tile, &profile_b_tile,
                                   last_dev);

  return SCAMP_NO_ERROR;
}

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices) {
  if (devices.empty()) {
    printf("Error: no gpu provided\n");
    exit(0);
  }
  // Allocate and initialize memory
  clock_t start, end;
  printf("%lf\n", args->distance_threshold());
  SCAMP_Operation op(args->timeseries_a().size(), args->timeseries_b().size(),
                     args->window(), args->max_tile_size(), devices,
                     !args->has_b(), args->precision_type(),
                     args->computing_columns() && args->computing_rows(),
                     args->distributed_start_row(),
                     args->distributed_start_col(), args->distance_threshold());
  op.init();
  gpuErrchk(cudaPeekAtLastError());
  start = clock();
  if (!args->has_b()) {
    op.do_join(args->timeseries_a(), args->timeseries_a(),
               args->mutable_profile_a(), args->mutable_profile_b());
  } else {
    op.do_join(args->timeseries_a(), args->timeseries_b(),
               args->mutable_profile_a(), args->mutable_profile_b());
  }
  cudaDeviceSynchronize();
  end = clock();
  gpuErrchk(cudaPeekAtLastError());
  op.destroy();
  gpuErrchk(cudaPeekAtLastError());
  printf(
      "Finished SCAMP to generate  matrix profile in %f "
      "seconds on %lu devices:\n",
      (end - start) / static_cast<double>(CLOCKS_PER_SEC), devices.size());
}

}  // namespace SCAMP
