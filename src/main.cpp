#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>
#include <limits>
#include <vector>
#include "SCAMP.h"
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
  SCAMP::FPtype t = SCAMP::FP_SINGLE;
  bool full_join = false;
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
        t = SCAMP::FP_DOUBLE;
        break;
      case 'm':
        t = SCAMP::FP_MIXED;
        break;
      case 'f':
        output_B_prefix = optarg;
        full_join = true;
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

  vector<uint32_t> profile(n_x, 0);
  vector<uint32_t> profile_B;

  if (full_join) {
    profile_B = vector<uint32_t>(n_y, 0);
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

  printf("Starting SCAMP\n");
  SCAMP::do_SCAMP(Ta_h, Tb_h, &profile, &profile_B, window_size, max_tile_size,
                  devices, self_join, t, full_join, start_row, start_col,
                  threshold);

  printf("Now writing result to files\n");
  FILE *f1 = fopen(argv[index++], "w");
  //  FILE *f2 = fopen(argv[index++], "w");
  FILE *f3, *f4;
  for (int i = 0; i < profile.size(); ++i) {
    fprintf(f1, "%u\n", profile[i]);
    //    fprintf(f1, "%f\n",
    //            sqrt(std::max(2.0 * window_size * (1.0 - profile[i]), 0.0)));
    //    fprintf(f2, "%u\n", profile_idx[i] + 1);
  }
  fclose(f1);
  //  fclose(f2);
  if (full_join) {
    f3 = fopen(strcat(output_B_prefix, "_mp"), "w");
    //    f4 = fopen(strcat(output_B_prefix, "i"), "w");
    for (int i = 0; i < profile_B.size(); ++i) {
      fprintf(f3, "%u\n", profile_B[i]);
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
