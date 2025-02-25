/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include "flash_mla.h"
#include "static_switch.h"

using namespace mcFlashAttn;

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    bool Is_splits = false,
    int kHeadDimV = kHeadDim
>
void run_flash_fwd_template(Flash_fwd_mla_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    bool Is_splits = false,
    int kHeadDimV = kHeadDim
>
void run_flash_splitkv_fwd_template(Flash_fwd_mla_params &params, cudaStream_t stream);

namespace mcFlashAttn {

    template<int Headdim>
    void run_mha_fwd_splitkv_dispatch(Flash_fwd_mla_params &params, const cudaStream_t stream);

    template<>
    inline void run_mha_fwd_splitkv_dispatch<576>(Flash_fwd_mla_params &params, const cudaStream_t stream) {
        Flash_launch_params launch_params;
        constexpr static int HeaddimQK = 576;
        constexpr static int HeaddimVO = 512;

        constexpr static int kBlockM = 32;
        constexpr static int kBlockN = 16;
        FP16_SWITCH(!params.is_bf16, [&] {
            BOOL_SWITCH(params.num_splits > 1, Is_splits, [&] {
                launch_params.block_type = 2;
                run_flash_splitkv_fwd_template<HeaddimQK, kBlockM, kBlockN, 2, true, true, elem_type, Is_splits, HeaddimVO>(params, stream);
            });
        });
    }

} // namespace mcFlashAttn end
