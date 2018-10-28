#pragma once
#include <float.h>
#include "SCAMP.pb.h"
#include "common.h"

namespace SCAMP {

SCAMPError_t kernel_self_join_upper(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    const cudaDeviceProp &props, SCAMPPrecisionType t, const OptionalArgs &args,
    SCAMPProfileType profile_type, cudaStream_t s);

SCAMPError_t kernel_self_join_lower(
    const double *QT, const double *df_A, const double *df_B,
    const double *dg_A, const double *dg_B, const double *norms_A,
    const double *norms_B, DeviceProfile *profile_A, DeviceProfile *profile_B,
    size_t window_size, size_t tile_width, size_t tile_height,
    size_t global_col, size_t global_row, const cudaDeviceProp &props,
    SCAMPPrecisionType t, const OptionalArgs &args,
    SCAMPProfileType profile_type, cudaStream_t s);

SCAMPError_t kernel_ab_join_upper(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    int64_t distributed_col, int64_t distributed_row,
    const cudaDeviceProp &props, SCAMPPrecisionType t, bool computing_rows,
    const OptionalArgs &args, SCAMPProfileType profile_type, cudaStream_t s);

SCAMPError_t kernel_ab_join_lower(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    int64_t distributed_col, int64_t distributed_row,
    const cudaDeviceProp &props, SCAMPPrecisionType t, bool computing_rows,
    const OptionalArgs &args, SCAMPProfileType profile_type, cudaStream_t s);

}  // namespace SCAMP
