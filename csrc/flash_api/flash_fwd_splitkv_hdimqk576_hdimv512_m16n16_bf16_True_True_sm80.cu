// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#include "flash_mla.h"
#include "flash_run_fwd_template_impl.h"
#include <mctlass/numeric_types.h>

template void run_flash_splitkv_fwd_template<
                576,
                16,
                16,
                4,
                true,
                true,
                cutlass::bfloat16_t,
                false,
                512,
                2
            >(Flash_fwd_mla_params &params, cudaStream_t stream);
