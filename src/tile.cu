#include "tile.h"
#include "kernels.h"
#include "fft_helper.h"
#include "common.h"

namespace SCRIMP {

SCRIMPError_t SCRIMP_Tile::do_self_join_full(cudaStream_t s) {
    SCRIMPError_t error;

    if(window_size > tile_width) {
        return SCRIMP_DIM_INCOMPATIBLE;
    }
    if(window_size > tile_height) { 
        return SCRIMP_DIM_INCOMPATIBLE;
    }
    error = fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, means_B, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }
    
    error = kernel_self_join_upper(QT_scratch, timeseries_A, timeseries_B, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A, profile_B, window_size, tile_width - window_size + 1, tile_height - window_size + 1, tile_start_A, tile_start_B, props, fp64, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    error = fft_info->compute_QT(QT_scratch, timeseries_B, timeseries_A, means_A, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }
    
    error = kernel_self_join_lower(QT_scratch, timeseries_A, timeseries_B, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A, profile_B, window_size, tile_width - window_size + 1, tile_height - window_size + 1, tile_start_A, tile_start_B, props, fp64, s);
    if(error != SCRIMP_NO_ERROR) {
        printf("SCRIMP error\n");
        return error;
    }

    return SCRIMP_NO_ERROR;

}

SCRIMPError_t SCRIMP_Tile::do_self_join_half(cudaStream_t s) {
    SCRIMPError_t error;

    if(window_size > tile_width) {
        return SCRIMP_DIM_INCOMPATIBLE;
    }
    if(window_size > tile_height) { 
        return SCRIMP_DIM_INCOMPATIBLE;
    }

    error = fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, means_B, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    error = kernel_self_join_upper(QT_scratch, timeseries_A, timeseries_B, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A, profile_B, window_size, tile_width - window_size + 1, tile_height - window_size + 1,tile_start_A, tile_start_B, props, fp64, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }
    
    return SCRIMP_NO_ERROR;
}

SCRIMPError_t SCRIMP_Tile::do_ab_join_full(cudaStream_t s) {
    SCRIMPError_t error;
    if(window_size > tile_width) {
        return SCRIMP_DIM_INCOMPATIBLE;
    }
    if(window_size > tile_height) { 
        return SCRIMP_DIM_INCOMPATIBLE;
    }
    error = fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, means_B, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }
    
    error = kernel_ab_join_upper(QT_scratch, timeseries_A, timeseries_B, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A, profile_B, window_size, tile_width - window_size + 1, tile_height - window_size + 1, tile_start_A, tile_start_B, props, fp64, full_join, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    error = fft_info->compute_QT(QT_scratch, timeseries_B, timeseries_A, means_A, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }
    
    error = kernel_ab_join_lower(QT_scratch, timeseries_A, timeseries_B, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A, profile_B, window_size, tile_width - window_size + 1, tile_height - window_size + 1, tile_start_A, tile_start_B, props, fp64, full_join, s);
    if(error != SCRIMP_NO_ERROR) {
        printf("SCRIMP error\n");
        return error;
    }

    return SCRIMP_NO_ERROR;
}

}

