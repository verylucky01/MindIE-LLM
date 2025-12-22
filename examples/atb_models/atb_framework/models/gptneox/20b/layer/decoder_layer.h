/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_MODELS_GPTNEOX_20B_PA_LAYER_H
#define ATB_SPEED_MODELS_GPTNEOX_20B_PA_LAYER_H

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"

namespace atb_speed {
namespace gptneox_20b {
struct PALayerParam {
    float layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    float rotaryPct = 0.0;
    float qScale = 1.0;
    float qkScale = 1.0;
    bool transposedWeight = true;
    std::string model = "gptneox_20b";
    bool isPrefill = false;
    int rank = 0;
    int rankSize = 1;
    std::string backend = "hccl";
};

enum LayerPATensorId : int {
    IN_HIDDENSTATES = 0,
    IN_INPUTLAYERNORMWEIGTH, // weights
    IN_INPUTLAYERNORMBIAS,
    IN_POSTATTNLAYERNORMWEIGHT,
    IN_POSTATTNLAYERNORMBIAS,
    IN_QKVWEIGHT,
    IN_QKVBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_FFNLINEARWEIGHT,
    IN_FFNLINEARBIAS,
    IN_FFNOUTLINEARWEIGHT,
    IN_FFNOUTLINEARBIAS,
    IN_POSITIONIDS, // inputs
    IN_COSEMBED,
    IN_SINEMBED,
    IN_ATTENTIONMASK,
    IN_CACHEK, // kvcache
    IN_CACHEV,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,

    OUT_GPTNEOXLAYEROUT,

    INTERMEDIATE_INPUTLAYERNORMOUT,
    INTERMEDIATE_MIXEDQKVLINEAROUT,
    INTERMEDIATE_QUERYEMBED,
    INTERMEDIATE_KEYEMBED,
    INTERMEDIATE_VALUE,
    INTERMEDIATE_QUERYEMBED_SCALED,
    INTERMEDIATE_SELFATTNOUT,
    INTERMEDIATE_SELFATTNLINEAROUT,
    INTERMEDIATE_POSTATTNLAYERNORMOUT,
    INTERMEDIATE_FFNLINEAROUT,
    INTERMEDIATE_FFNACTOUT,
    INTERMEDIATE_FFNOUTLINEAROUT,
    INTERMEDIATE_ATTNMLPADDOUT,
    INTERMEDIATE_ATTNMLP_ALLREDUCEOUT
};

struct PositionEmbeddingPAParam {
    int32_t headNum = 0;
    int32_t dk = 0;
    float rotaryPct = 0.25;
};

atb::Status PositionEmbeddingPAOperation(const PositionEmbeddingPAParam &param, atb::Operation **operation);

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation);

atb::Operation *CreatePALayer(const nlohmann::json &paramJson);
} // namespace gptneox_20b
} // namespace atb_speed

#endif // ATB_SPEED_MODELS_GPTNEOX_20B_PA_LAYER_H
