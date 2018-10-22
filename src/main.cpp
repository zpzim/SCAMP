#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>
#include <limits>
#include <vector>
#include "SCAMP.h"
#include "SCAMP.pb.h"
#include "common.h"

// Reads input time series from file
template <class DTYPE>
void readFile(const char *filename, vector<DTYPE> &v, const char *format_str) {
  FILE *f = fopen(filename, "r");
  if (f == NULL) {
    printf("Unable to open %s for reading, please make sure it exists\n",
           filename);
    exit(0);
  }
  DTYPE num;
  while (!feof(f)) {
    fscanf(f, format_str, &num);
    v.push_back(num);
  }
  v.pop_back();
  fclose(f);
}

int main(int argc, char **argv) {
  SCAMP::SCAMPPrecisionType t = SCAMP::PRECISION_SINGLE;
  bool full_join = false;
  bool computing_columns = true;
  bool computing_rows = true;
  bool self_join = true;
  size_t start_row = 0;
  size_t start_col = 0;
  int max_tile_size = (1 << 20);
  int opt;
  std::vector<int> devices;
  std::vector<double> Ta_h, Tb_h;
  char *output_B_prefix, *input_B;
  while ((opt = getopt(argc, argv, "mdf:r:c:s:b:g:")) != -1) {
    switch (opt) {
      case 'd':
        t = SCAMP::PRECISION_DOUBLE;
        break;
      case 'm':
        t = SCAMP::PRECISION_MIXED;
        break;
      case 'f':
        output_B_prefix = optarg;
        full_join = true;
        computing_rows = true;
        computing_columns = true;
        break;
      case 'r':
        start_row = atoi(optarg);
        break;
      case 'c':
        start_col = atoi(optarg);
        break;
      case 's':
        max_tile_size = atoi(optarg);
        break;
      case 'b':
        input_B = optarg;
        self_join = false;
        computing_columns = true;
        computing_rows = false;
        break;
      case 'g':
        devices.push_back(atoi(optarg));
        break;
      default:
        exit(EXIT_FAILURE);
    }
  }
  if (self_join && full_join) {
    printf(
        "error: invalid argument combination -f flag can only be used in "
        "ab-joins");
    exit(EXIT_FAILURE);
  }

  int index = optind;
  int window_size = atoi(argv[index++]);
  float threshold = atof(argv[index++]);
  printf("%lf\n", static_cast<double>(threshold));
  char *input_A = argv[index++];

  readFile<double>(input_A, Ta_h, "%lf");

  if (!self_join) {
    readFile<double>(input_B, Tb_h, "%lf");
  }

  int n_x = Ta_h.size() - window_size + 1;
  int n_y;
  if (self_join) {
    n_y = n_x;
  } else {
    n_y = Tb_h.size() - window_size + 1;
  }

  if (devices.empty()) {
    // Use all available devices
    printf("using all devices\n");
    int num_dev;
    cudaGetDeviceCount(&num_dev);
    for (int i = 0; i < num_dev; ++i) {
      devices.push_back(i);
    }
  }
  SCAMP::SCAMPArgs args;
  args.set_window(window_size);
  args.set_max_tile_size(max_tile_size);
  args.set_has_b(!self_join);
  args.set_distributed_start_row(start_row);
  args.set_distributed_start_col(start_col);
  args.set_distance_threshold(static_cast<double>(threshold));
  args.set_computing_columns(computing_columns);
  args.set_computing_rows(computing_rows);
  args.mutable_profile_a()->set_type(SCAMP::PROFILE_TYPE_SUM_THRESH);
  args.mutable_profile_b()->set_type(SCAMP::PROFILE_TYPE_SUM_THRESH);

  args.set_precision_type(t);
  {
    google::protobuf::RepeatedField<double> data(Ta_h.begin(), Ta_h.end());
    args.mutable_timeseries_a()->Swap(&data);
    data = google::protobuf::RepeatedField<double>(Tb_h.begin(), Tb_h.end());
    args.mutable_timeseries_b()->Swap(&data);
  }
  vector<double> temp(n_x, 0);
  {
    google::protobuf::RepeatedField<double> data(temp.begin(), temp.end());
    args.mutable_profile_a()
        ->mutable_data()
        ->Add()
        ->mutable_double_value()
        ->mutable_value()
        ->Swap(&data);
  }
  if (full_join) {
    temp.resize(n_y, 0);
    google::protobuf::RepeatedField<double> data(temp.begin(), temp.end());
    args.mutable_profile_b()
        ->mutable_data()
        ->Add()
        ->mutable_double_value()
        ->mutable_value()
        ->Swap(&data);
  }

  printf("Starting SCAMP\n");
  SCAMP::do_SCAMP(&args, devices);

  printf("Now writing result to files\n");
  FILE *f1 = fopen(argv[index++], "w");
  //  FILE *f2 = fopen(argv[index++], "w");
  FILE *f3, *f4;
  for (int i = 0; i < n_x; ++i) {
    fprintf(f1, "%lf\n",
            args.profile_a().data().Get(0).double_value().value().Get(i));
    //    fprintf(f1, "%f\n",
    //            sqrt(std::max(2.0 * window_size * (1.0 - profile[i]), 0.0)));
    //    fprintf(f2, "%u\n", profile_idx[i] + 1);
  }
  fclose(f1);
  //  fclose(f2);
  if (full_join) {
    f3 = fopen(strcat(output_B_prefix, "_mp"), "w");
    //    f4 = fopen(strcat(output_B_prefix, "i"), "w");
    for (int i = 0; i < n_y; ++i) {
      fprintf(f3, "%lf\n",
              args.profile_b().data().Get(0).double_value().value().Get(i));
      //      fprintf(f3, "%f\n",
      //              sqrt(std::max(2.0 * window_size * (1.0 - profile_B[i]),
      //              0.0)));
      //      fprintf(f4, "%u\n", profile_idx_B[i] + 1);
    }
  }
  if (full_join) {
    fclose(f3);
    //    fclose(f4);
  }
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaDeviceReset());
  printf("Done\n");
  return 0;
}
