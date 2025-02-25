#pragma once

#include <cuda.h>

#include "flash_mla.h"
#include "static_switch.h"
#include "flash_fwd_split_kernel_k64_V1x8.h"
#include "feature/attn_mask.h"

using namespace mcFlashAttn;

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Is_page_attn>
__global__ void flash_fwd_splitkv_kernel(const Flash_fwd_mla_params params, const int num_m_block) {
    flash::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Is_page_attn>(params, num_m_block);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_mla_params &params, cudaStream_t stream) {

    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH_AND_CONST_PRECOND((!Is_causal), (params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                BOOL_SWITCH(params.block_table != nullptr, Is_page_attn, [&] {
                    auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, false, false, false, false, false, false, Is_page_attn>;
                    if (smem_size >= 32 * 1024) {
                        CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params, num_m_block);
                    CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}
