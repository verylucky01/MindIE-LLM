/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册add_custom和npu_selected_flash_attention和后续的XXX算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(mie_ops, m) {
    m.def("npu_mla_process(Tensor input, Tensor gamma0, Tensor beta0, Tensor wdqkv, Tensor descale0, Tensor gamma1, Tensor beta1, Tensor wuq, Tensor descale1, Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache, Tensor kv_cache_rope, Tensor slotmapping, *, Tensor? quant_scale0=None, Tensor? quant_offset0=None, Tensor? bias0=None, Tensor? quant_scale1=None, Tensor? quant_offset1=None, Tensor? bias1=None, Tensor? ctkv_scale=None, Tensor? q_nope_scale=None, str? cache_mode_opt=None, str? quant_mode_opt=None) -> (Tensor, Tensor, Tensor, Tensor)");
}
