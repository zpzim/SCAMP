#include "scamp_interface.h"

#include "core/SCAMP.h"
#include "scamp_exception.h"

#include <thread>
#include <vector>

namespace SCAMP {

int num_available_gpus() {
  int num_dev = 0;
#ifdef _HAS_CUDA_
  cudaGetDeviceCount(&num_dev);
#endif
  return num_dev;
}

void do_SCAMP(SCAMPArgs *args) {
  int num_threads = 0;
  int num_devices = num_available_gpus();
  std::vector<int> devices(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    devices.at(i) = i;
  }
  if (devices.empty()) {
    num_threads = std::thread::hardware_concurrency();
  }
  do_SCAMP(args, devices, num_threads);
}

// Wrapper on SCAMP_Operation called by main(), this function constructs
// and executes a SCAMP_Operation given a set of user selected parameters.
void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices,
              int num_threads) {
  if (devices.empty() && num_threads <= 0) {
    throw SCAMPException("Error: no compute_resources provided");
  }

  if (args == nullptr) {
    throw SCAMPException("Error: Invalid arguments provided to SCAMP");
  }

  if (!args->silent_mode) {
    std::cout << "Validating SCAMP args." << std::endl;
  }
  args->validate();

  // Allocate and initialize memory
  if (!args->InitProfileMemory()) {
    throw SCAMPException("Error: Invalid arguments provided to SCAMP");
  }

  OptionalArgs _opt_args(args->distance_threshold);

  if (!args->silent_mode) {
    std::cout << "Building SCAMP Operation from args" << std::endl;
  }

  // Construct operation
  SCAMP_Operation op(
      args->timeseries_a.size(), args->timeseries_b.size(), args->window,
      args->max_tile_size, devices, !args->has_b, args->precision_type,
      args->distributed_start_row, args->distributed_start_col, _opt_args,
      args->profile_type, &args->profile_a, &args->profile_b,
      args->keep_rows_separate, args->computing_rows, args->computing_columns,
      args->is_aligned, args->silent_mode, num_threads,
      args->max_matches_per_column, args->matrix_height, args->matrix_width);

  if (!args->silent_mode) {
    std::cout << "SCAMP Operation constructed" << std::endl;
  }
  // Execute op
  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  if (args->has_b) {
    op.do_join(args->timeseries_a, args->timeseries_b);
  } else {
    op.do_join(args->timeseries_a, args->timeseries_a);
  }
  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  if (!args->silent_mode) {
    printf(
        "Finished %d SCAMP tiles to generate  matrix profile in %lf "
        "seconds on %lu devices and %d threads\n",
        op.get_completed_tiles(),
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            static_cast<double>(1000000),
        devices.size(), num_threads);
  }
}

}  // namespace SCAMP
