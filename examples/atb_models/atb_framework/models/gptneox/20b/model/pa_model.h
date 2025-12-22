/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_MODELS_GPTNEOX_20B_PA_MODEL_H
#define ATB_SPEED_MODELS_GPTNEOX_20B_PA_MODEL_H

#include "atb_speed/base/model.h"

namespace atb_speed {
namespace gptneox_20b {

enum InTensorId : uint32_t {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_TENSOR_PLACE_HOLDER,
    IN_TENSOR_MAX
};

enum InternalTensorId : uint32_t {
    INTERNAL_TENSOR_ID = 0,
    INTERNAL_TENSOR_MAX
};

enum OutTensorId : uint32_t {
    OUT_TENSOR_ID = 0,
    OUT_TENSOR_MAX,
};

class PAModel : public Model {
public:
    struct Param {
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        float layerNormEps = 0;
        float rotaryPct = 0.0;
        int rank = 0;
        int rankSize = 1;
        bool isPrefill = false;
        float qScale = 1.0;
        float qkScale = 1.0;
        std::string backend = "hccl";

        void FromString(const std::string &param);
    };

    explicit PAModel(const std::string &param);

    ~PAModel() override;

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
        std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    int64_t AddWordEmbedding();
    int64_t AddLayer();
    int64_t AddFinalNorm();
    int64_t AddLmhead();
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;

private:
    Param param_;
    std::vector<int32_t> seqLen_;
};

} // namespace gptneox_20b
} // namespace atb_speed
#endif // ATB_SPEED_MODELS_GPTNEOX_20B_PA_MODEL_H
