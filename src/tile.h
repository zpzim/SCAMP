#pragma once

#include "common.h"
#include "fft_helper.h"
#include "kernels.h"

namespace SCRIMP {

template <class DATATYPE, class CUFFT_DATATYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
class SCRIMP_Tile {
private:
    SCRIMPTileType type;
    const DATATYPE *timeseries_A;
    const DATATYPE *timeseries_B;   
    const DATATYPE *means_A;
    const DATATYPE *means_B;
    const DATATYPE *std_dev_A;
    const DATATYPE *std_dev_B;
    fft_precompute_helper<DATATYPE, CUFFT_DATATYPE> *fft_info;
    DATATYPE *QT_scratch;
    unsigned long long int *profile_A;
    unsigned long long int *profile_B;
    
    size_t tile_start_A;
    size_t tile_start_B;
    size_t tile_height;
    size_t tile_width;
    const size_t window_size;

    SCRIMPError_t do_self_join_full(cudaStream_t s);
    SCRIMPError_t do_self_join_half(cudaStream_t s);
    SCRIMPError_t do_ab_join_full(cudaStream_t s);
    SCRIMPError_t do_ab_join_upper(cudaStream_t s);
    SCRIMPError_t do_ab_join_lower(cudaStream_t s);

public:
    SCRIMP_Tile(SCRIMPTileType t, const DATATYPE *ts_A, const DATATYPE *ts_B, const DATATYPE *mu_A,
                const DATATYPE *mu_B, const DATATYPE *sigma_A, const DATATYPE *sigma_B, 
                DATATYPE *QT, unsigned long long int *profileA, unsigned long long int *profileB,
                size_t start_A, size_t start_B, size_t height, size_t width, size_t m,  
                fft_precompute_helper<DATATYPE, CUFFT_DATATYPE> *scratch)
                : type(t), timeseries_A(ts_A), timeseries_B(ts_B), means_A(mu_A), means_B(mu_B),
                  std_dev_A(sigma_A), std_dev_B(sigma_B), QT_scratch(QT), profile_A(profileA),
                  profile_B(profileB), tile_start_A(start_A), tile_start_B(start_B), tile_height(height),
                  tile_width(width), fft_info(scratch), window_size(m) {}
    SCRIMPError_t execute(cudaStream_t s) {
        SCRIMPError_t error;
        switch (type) {
            case SELF_JOIN_FULL_TILE:
                error = do_self_join_full(s);
                break;
            case SELF_JOIN_UPPER_TRIANGULAR:
                error = do_self_join_half(s);
                break;
            case AB_JOIN_FULL_TILE:
                error = do_ab_join_full(s);
                break;
            case AB_JOIN_UPPER_TRIANGULAR:
                error = do_ab_join_upper(s);
                break;
            case AB_JOIN_LOWER_TRIANGULAR:
                error = do_ab_join_lower(s);
                break;
            default:
                error = SCRIMP_TILE_ILLEGAL_TYPE;
                break;
        }
        return error;
    }
};


template<class DATATYPE, class CUFFT_DTYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t SCRIMP_Tile<DATATYPE, CUFFT_DTYPE, BLOCKSZ, UNROLL_COUNT>::do_self_join_full(cudaStream_t s) {
    SCRIMPError_t error;

    if(window_size > tile_width) {
        return SCRIMP_DIM_INCOMPATIBLE;
    }
    if(window_size > tile_height) { 
        return SCRIMP_DIM_INCOMPATIBLE;
    }
    error = fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }
    
    error = kernel_self_join_upper<DATATYPE, BLOCKSZ, UNROLL_COUNT>(QT_scratch, timeseries_A, timeseries_B, std_dev_A, std_dev_B, means_A, means_B, profile_A, profile_B, window_size, tile_width - window_size + 1, tile_start_A, tile_start_B, s, false);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    error = fft_info->compute_QT(QT_scratch, timeseries_B, timeseries_A, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }
    
    error = kernel_self_join_lower<DATATYPE, BLOCKSZ, UNROLL_COUNT>(QT_scratch, timeseries_A, timeseries_B, std_dev_A, std_dev_B, means_A, means_B, profile_A, profile_B, window_size, tile_width - window_size + 1, tile_height - window_size + 1, tile_start_A, tile_start_B, s);
    if(error != SCRIMP_NO_ERROR) {
        printf("SCRIMP error\n");
        return error;
    }

    return SCRIMP_NO_ERROR;

}

template<class DATATYPE, class CUFFT_DTYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t SCRIMP_Tile<DATATYPE, CUFFT_DTYPE, BLOCKSZ, UNROLL_COUNT>::do_self_join_half(cudaStream_t s) {
    SCRIMPError_t error;

    if(window_size > tile_width) {
        return SCRIMP_DIM_INCOMPATIBLE;
    }
    if(window_size > tile_height) { 
        return SCRIMP_DIM_INCOMPATIBLE;
    }

    error = fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    error = kernel_self_join_upper<DATATYPE, BLOCKSZ, UNROLL_COUNT>(QT_scratch, timeseries_A, timeseries_B, std_dev_A, std_dev_B, means_A, means_B, profile_A, profile_B, window_size, tile_width - window_size + 1, tile_start_A, tile_start_B, s, false);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }
    
    return SCRIMP_NO_ERROR;
}

template<class DATATYPE, class CUFFT_DTYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t SCRIMP_Tile<DATATYPE, CUFFT_DTYPE, BLOCKSZ, UNROLL_COUNT>::do_ab_join_full(cudaStream_t s) {
    return SCRIMP_FUNCTIONALITY_UNIMPLEMENTED;
}

template<class DATATYPE, class CUFFT_DTYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t SCRIMP_Tile<DATATYPE, CUFFT_DTYPE, BLOCKSZ, UNROLL_COUNT>::do_ab_join_upper(cudaStream_t s) {
    return SCRIMP_FUNCTIONALITY_UNIMPLEMENTED;
}

template<class DATATYPE, class CUFFT_DTYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t SCRIMP_Tile<DATATYPE, CUFFT_DTYPE, BLOCKSZ, UNROLL_COUNT>::do_ab_join_lower(cudaStream_t s) {
    return SCRIMP_FUNCTIONALITY_UNIMPLEMENTED;
}
}
