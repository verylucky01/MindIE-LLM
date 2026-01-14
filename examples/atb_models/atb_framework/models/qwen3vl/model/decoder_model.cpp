/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "models/qwen3vl/model/decoder_model.h"
#include "operations/fusion/infer_shape_functions.h"

namespace atb_speed {
namespace qwen3vl {

DecoderModel::DecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->inTensorCandidates["deepstack_visual_embeds"] = {
        "deepstack_visual_embeds_0", "deepstack_visual_embeds_1", "deepstack_visual_embeds_2"};
}

void DecoderModel::ConstructInTensorMap()
{
    atb_speed::base::DecoderModel::ConstructInTensorMap();
    if (this->param.isPrefill) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "deepstack_visual_embeds", this->inTensorMap);
    }
}

void DecoderModel::SetLayerNodeOptionalInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    atb_speed::base::DecoderModel::SetLayerNodeOptionalInput(layerNode, layerId, inTensorId);
    if (this->param.isPrefill) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "deepstack_visual_embeds_0"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "deepstack_visual_embeds_1"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "deepstack_visual_embeds_2"));
    }
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::base::LayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    DecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

} // namespace qwen3vl
} // namespace atb_speed