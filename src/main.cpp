#ifdef _HAS_CUDA_
#include <cuda_runtime.h>
#endif

#include <gflags/gflags.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "common/common.h"
#include "common/scamp_args.h"
#include "common/scamp_exception.h"
#include "common/scamp_utils.h"
#ifdef _DISTRIBUTED_EXECUTION_
#include <distributed/scamp_interface.h>
#else
#include <common/scamp_interface.h>
#endif

#ifdef _DISTRIBUTED_EXECUTION_
DEFINE_int64(distributed_tile_size, 4000000,
             "tile size to use for computation on worker notes");
DEFINE_string(hostname_port, "localhost:30078",
              "Hostname:Port of SCAMP server to perform distributed work");
#endif
DEFINE_int64(max_matches_per_column, 5,
             "Maximum number of neighbors to generate for any subsequence "
             "(used for ALL_NEIGHBORS profiles).");
DEFINE_int32(reduced_height, 50, "The final height of the output matrix");
DEFINE_int32(reduced_width, 50, "The final width of the output matrix");
DEFINE_bool(print_debug_info, false, "Whether SCAMP will print debug info.");
DEFINE_int32(num_cpu_workers, 0, "Number of CPU workers to use");
DEFINE_bool(output_pearson, false,
            "If true SCAMP will output pearson correlation instead of "
            "z-normalized euclidean distance.");
DEFINE_bool(
    no_gpu, false,
    "If true SCAMP will not use any GPUs to compute the matrix profile");
DEFINE_int32(max_tile_size, 1 << 17, "Maximum tile size SCAMP will use");
DEFINE_int32(window, -1, "Length of subsequences to search for");
DEFINE_double(
    threshold, 0,
    "Distance threshold for frequency and sum calculations, we will only count "
    "events with a Pearson correlation above this threshold.");
DEFINE_string(
    profile_type, "1NN_INDEX",
    "Matrix Profile Type to compute, must be one of \"1NN_INDEX, SUM_THRESH\", "
    "1NN_INDEX generates the classic Matrix Profile, SUM_THRESH generates a "
    "sum of the correlations above threshold set by the --threshold flag.");
DEFINE_bool(ultra_precision, false,
            "Ultra high precision computation with a potential performance hit "
            "(for large subsequence lengths).");
DEFINE_bool(double_precision, false, "Computation in double precision");
DEFINE_bool(mixed_precision, false, "Computation in mixed precision");
DEFINE_bool(single_precision, false, "Computation in single precision");
DEFINE_bool(
    keep_rows, false,
    "Informs SCAMP to compute the \"rowwise mp\" and output in a a separate "
    "file specified by the flag --output_b_file_name, only valid for ab-joins, "
    "this is useful when computing a distributed self-join, so as to not "
    "recompute values in the lower-trianglular portion of the symmetric "
    "distance matrix.");
DEFINE_bool(aligned, false,
            "For ab-joins which are partially self-joins. And for distributed "
            "self-joins. Indicates that A and B may start with the same "
            "sequence and must consider an exclusion zone");
DEFINE_int64(
    global_row, -1,
    "Informs SCAMP that this join is part of a larger distributed join which "
    "starts at this row in the larger distance matrix, this allows us to pick "
    "an appropriate exclusion zone for our computation if necessary.");
DEFINE_int64(
    global_col, -1,
    "Informs SCAMP that this join is part of a larger distributed join which "
    "starts at this column in the larger distance matrix, this allows us to "
    "pick an appropriate exclusion zone for our computation if necessary");
DEFINE_string(input_a_file_name, "",
              "Primary input file name for a self-join or ab-join");
DEFINE_string(input_b_file_name, "",
              "Secondary input file name for an ab-join");
DEFINE_string(output_a_file_name, "mp_columns_out",
              "Primary output file name for the matrix profile \"columns\"");
DEFINE_string(
    output_a_index_file_name, "mp_columns_out_index",
    "Primary output file name for the matrix profile \"columns\" index (if "
    "ab-join these are indexes into input_b) this flag is only used when "
    "generating a matrix profile which contains an index");
DEFINE_string(output_b_file_name, "mp_rows_out",
              "Output the matrix profile for the \"rows\" as a separate file "
              "with this name");
DEFINE_string(
    output_b_index_file_name, "mp_rows_out_index",
    "Primary output file name for the matrix profile \"columns\" index (if "
    "ab-join these are indexes into input_a) this flag is only used when "
    "generating a matrix profile which contains an index");
DEFINE_string(gpus, "",
              "IDs of GPUs on the system to use, if this flag is not set SCAMP "
              "tries to use all available GPUs on the system");

int main(int argc, char **argv) {
  bool self_join, computing_rows, computing_cols;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (!FLAGS_ultra_precision && !FLAGS_double_precision &&
      !FLAGS_mixed_precision && !FLAGS_single_precision) {
    FLAGS_double_precision = true;
  }
  if ((FLAGS_ultra_precision ? 1 : 0) + (FLAGS_double_precision ? 1 : 0) +
          (FLAGS_mixed_precision ? 1 : 0) + (FLAGS_single_precision ? 1 : 0) !=
      1) {
    printf("Error: only one precision flag can be enabled at a time\n");
    return 1;
  }
  if (FLAGS_window < 3) {
    printf(
        "Error: Subsequence length must be at least 3, please use "
        "--window=<window_size> to specify your subsequence length.\n");
    return 1;
  }
  if (FLAGS_max_tile_size < 1024) {
    printf("Error: max tile size must be at least 1024\n");
    return 1;
  }
  if (FLAGS_max_tile_size / 2 < FLAGS_window) {
    printf(
        "Error: Tile length and width must be at least 2x larger than the "
        "window size. Please set a larger --max_tile_size=<max_tile_size>\n");
    return 1;
  }
  std::vector<int> devices = ParseIntList(FLAGS_gpus);
  if (FLAGS_input_a_file_name.empty()) {
    printf(
        "Error: primary input filename must be specified using "
        "--input_a_file_name");
    return 1;
  }
  if (FLAGS_input_b_file_name.empty()) {
    self_join = true;
    computing_rows = true;
    computing_cols = true;
  } else {
    self_join = false;
    computing_cols = true;
    computing_rows = FLAGS_keep_rows;
  }

  SCAMP::SCAMPPrecisionType t =
      GetPrecisionType(FLAGS_ultra_precision, FLAGS_double_precision,
                       FLAGS_mixed_precision, FLAGS_single_precision);
  SCAMP::SCAMPProfileType profile_type = ParseProfileType(FLAGS_profile_type);

  std::vector<double> Ta_h, Tb_h;

  readFile(FLAGS_input_a_file_name, Ta_h);

  if (!self_join) {
    readFile(FLAGS_input_b_file_name, Tb_h);
  }

  int n_x = Ta_h.size() - FLAGS_window + 1;
  int n_y;
  if (self_join) {
    n_y = n_x;
  } else {
    n_y = Tb_h.size() - FLAGS_window + 1;
  }
  if (n_x < 1 || n_y < 1) {
    printf("Error: window size must be smaller than the timeseries length\n");
    return 1;
  }

#ifdef _HAS_CUDA_
  if (devices.empty() && !FLAGS_no_gpu) {
    // Use all available devices
    if (FLAGS_print_debug_info) {
      printf("using all devices\n");
    }
    int num_dev;
    cudaGetDeviceCount(&num_dev);
    for (int i = 0; i < num_dev; ++i) {
      devices.push_back(i);
    }
  }
#else
  // We cannot use gpus if we don't have CUDA
  ASSERT(devices.empty(),
         "This binary was not built with CUDA, --gpus cannot be used with this "
         "binary.");
#endif
  SCAMP::SCAMPArgs args;
  args.window = FLAGS_window;
  args.max_tile_size = FLAGS_max_tile_size;
  args.has_b = !self_join;
  args.distributed_start_row = FLAGS_global_row;
  args.distributed_start_col = FLAGS_global_col;
  args.distance_threshold = static_cast<double>(FLAGS_threshold);
  args.computing_columns = computing_cols;
  args.computing_rows = computing_rows;
  args.profile_a.type = profile_type;
  args.profile_b.type = profile_type;
  args.precision_type = t;
  args.profile_type = profile_type;
  args.keep_rows_separate = FLAGS_keep_rows;
  args.is_aligned = FLAGS_aligned;
  args.timeseries_a = std::move(Ta_h);
  args.timeseries_b = std::move(Tb_h);
  args.silent_mode = !FLAGS_print_debug_info;
  args.max_matches_per_column = FLAGS_max_matches_per_column;
  args.matrix_height = FLAGS_reduced_height;
  args.matrix_width = FLAGS_reduced_width;
  if (FLAGS_print_debug_info) {
    printf("Starting SCAMP\n");
  }
  try {
#ifdef _DISTRIBUTED_EXECUTION_
    do_SCAMP_distributed(&args, FLAGS_hostname_port,
                         FLAGS_distributed_tile_size);
#else
    SCAMP::do_SCAMP(&args, devices, FLAGS_num_cpu_workers);
#endif
  } catch (const SCAMPException &e) {
    std::cout << e.what() << "\n";
    exit(1);
  }
  if (FLAGS_print_debug_info) {
    printf("Now writing result to files\n");
  }
  WriteProfileToFile(FLAGS_output_a_file_name, FLAGS_output_a_index_file_name,
                     args.profile_a, FLAGS_output_pearson, FLAGS_window,
                     FLAGS_reduced_width, FLAGS_reduced_height);
  if (FLAGS_keep_rows) {
    WriteProfileToFile(FLAGS_output_b_file_name, FLAGS_output_b_index_file_name,
                       args.profile_b, FLAGS_output_pearson, FLAGS_window,
                       FLAGS_reduced_width, FLAGS_reduced_height);
  }
  if (FLAGS_print_debug_info) {
    printf("Done\n");
  }
  return 0;
}
