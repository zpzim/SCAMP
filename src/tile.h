#pragma once

#include "common.h"
#include "fft_helper.h"
#include "kernels.h"

namespace SCRIMP {

class SCRIMP_Tile {
private:
    SCRIMPTileType type;
    const double *timeseries_A;
    const double *timeseries_B;   
    const double *df_A;
    const double *df_B;
    const double *dg_A;
    const double *dg_B;
    const double *means_A;
    const double *means_B;
    const double *norms_A;
    const double *norms_B;
    fft_precompute_helper *fft_info;
    double *QT_scratch;
    unsigned long long int *profile_A;
    unsigned long long int *profile_B;
    
    size_t tile_start_A;
    size_t tile_start_B;
    size_t tile_height;
    size_t tile_width;
    const size_t window_size;
    bool full_join;
    const bool fp64;
    const cudaDeviceProp props;

    SCRIMPError_t do_self_join_full(cudaStream_t s);
    SCRIMPError_t do_self_join_half(cudaStream_t s);
    SCRIMPError_t do_ab_join_full(cudaStream_t s);
    SCRIMPError_t do_ab_join_upper(cudaStream_t s);
    SCRIMPError_t do_ab_join_lower(cudaStream_t s);

public:
    SCRIMP_Tile(SCRIMPTileType t, const double *ts_A, const double *ts_B, const double *dfA,
                const double *dfB, const double *dgA, const double *dgB, const double *normA,
                const double *normB, const double *meansA, const double *meansB, double *QT,
                unsigned long long int *profileA, unsigned long long int *profileB,
                size_t start_A, size_t start_B, size_t height, size_t width, size_t m,
                fft_precompute_helper *scratch, const cudaDeviceProp &prop, bool use_double)
                : type(t), timeseries_A(ts_A), timeseries_B(ts_B), df_A(dfA), df_B(dfB), means_A(meansA), means_B(meansB),
                  dg_A(dgA), dg_B(dgB), norms_A(normA), norms_B(normB), QT_scratch(QT), profile_A(profileA),
                  profile_B(profileB), tile_start_A(start_A), tile_start_B(start_B), tile_height(height),
                  tile_width(width), fft_info(scratch), window_size(m), props(prop), fp64(use_double), full_join(false) {}
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
            case AB_FULL_JOIN_FULL_TILE:
                full_join = true;
                error = do_ab_join_full(s);
                break;
            default:
                error = SCRIMP_TILE_ILLEGAL_TYPE;
                break;
        }
        return error;
    }
};


}
