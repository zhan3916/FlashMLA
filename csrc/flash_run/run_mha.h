#pragma once

#include "flash_mla.h"

void run_mha_fwd(mcFlashAttn::Flash_fwd_mla_params &params, cudaStream_t stream, bool force_split_kernel=false);
