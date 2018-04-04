#include "tile.h"
#include "kernels.h"
#include "fft_helper.h"
#include "common.h"

namespace SCRIMP {


template<class DATATYPE, class CUFFT_DTYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t SCRIMP_Tile<DATATYPE, CUFFT_DTYPE, BLOCKSZ, UNROLL_COUNT>::do_self_join_full(cudaStream_t s) {
    SCRIMPError_t error;

    error = fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, window_size, tile_width, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    error = kernel_self_join_upper<DATATYPE, BLOCKSZ, UNROLL_COUNT>(QT_scratch, timeseries_A, timeseries_B, std_dev_A, std_dev_B, means_A, means_B, profile_A, profile_B, window_size, tile_width, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    error = fft_info->compute_QT(QT_scratch, timeseries_B, timeseries_A, window_size, tile_width, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    error = kernel_self_join_lower<DATATYPE, BLOCKSZ, UNROLL_COUNT>(QT_scratch, timeseries_A, timeseries_B, std_dev_A, std_dev_B, means_A, means_B, profile_A, profile_B, window_size, tile_width, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    return SCRIMP_NO_ERROR;

}

template<class DATATYPE, class CUFFT_DTYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t SCRIMP_Tile<DATATYPE, CUFFT_DTYPE, BLOCKSZ, UNROLL_COUNT>::do_self_join_half(cudaStream_t s) {
    SCRIMPError_t error;

    error = fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, window_size, tile_width, s);
    if(error != SCRIMP_NO_ERROR) {
        return error;
    }

    error = kernel_self_join_upper<DATATYPE, BLOCKSZ, UNROLL_COUNT>(QT_scratch, timeseries_A, timeseries_B, std_dev_A, std_dev_B, means_A, means_B, profile_A, profile_B, window_size, tile_width, s);
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


/*
template<class DATATYPE, class CUFFT_DTYPE, size_t BLOCKSZ, size_t UNROLL_COUNT>
SCRIMPError_t SCRIMP_Tile<DATATYPE, CUFFT_DTYPE, BLOCKSZ, UNROLL_COUNT>::execute(cudaStream_t s) {
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
*/

}

