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
#include "gelu_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "system_log.h"
#include "aclnnop/aclnn_gelu.h"
#include "aclnnop/aclnn_gelu_v2.h"

namespace atb_speed::common {

    GeluOperation::GeluOperation(
        const std::string &name,
        atb_speed::common::AclNNGeluParam param
    ) : AclNNOperation(name), param_(param)
    {
        this->opName_ = name;
        this->param_ = param;
    }

    GeluOperation::~GeluOperation()
    {
        LOG_DEBUG_MODEL << "GeluOperation deconstruct";
        this->DestroyOperation();
    }

    /**
     *
     * @param[in] inTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @param[in] outTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @return atb::Status
     */
    atb::Status GeluOperation::InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc,
        atb::SVector<atb::TensorDesc> &outTensorDesc
    ) const
    {
        LOG_DEBUG_MODEL << opName_ << " InferShape start";
        outTensorDesc.at(0).format = inTensorDesc.at(0).format;
        outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
        outTensorDesc.at(0).shape.dimNum = inTensorDesc.at(0).shape.dimNum;

        LOG_DEBUG_MODEL << "Check " << opName_ << " input dimNum=" << inTensorDesc.at(0).shape.dimNum;
        for (uint64_t dim = 0; dim < inTensorDesc.at(0).shape.dimNum; ++dim) {
            LOG_DEBUG_MODEL << "input dim" << dim << " shape=" << inTensorDesc.at(0).shape.dims[dim];
            outTensorDesc.at(0).shape.dims[dim] = inTensorDesc.at(0).shape.dims[dim];
        }

        LOG_DEBUG_MODEL << opName_ << " InferShape end";
        return atb::NO_ERROR;
    }

    uint32_t GeluOperation::GetInputNum() const
    {
        return NUM1;  // inputTensorNum = 1
    }

    uint32_t GeluOperation::GetOutputNum() const
    {
        return NUM1;  // outputTensorNum = 1
    }

    atb::Status GeluOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclInTensors.resize(GetInputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
            LOG_DEBUG_MODEL << opName_ << " CreateTensor start";
            atb::Tensor (*ReshapeTensorDecsFuncPtr)(atb::Tensor) = &SqueezeBatchSeq;
            if (CreateTensor(variantPack.inTensors.at(i), i, aclnnVariantPack.aclInTensors[i],
                             ReshapeTensorDecsFuncPtr) != atb::NO_ERROR) {
                LOG_ERROR_MODEL << this->opName_ << " InTensor aclCreateTensor index " << i << " fail";
                return atb::ERROR_INTERNAL_ERROR;
            }
            LOG_DEBUG_MODEL << opName_ << " CreateTensor end";
        }
        return atb::NO_ERROR;
    }

    atb::Status GeluOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
            LOG_DEBUG_MODEL << opName_ << " CreateTensor start";
            atb::Tensor (*ReshapeTensorDecsFuncPtr)(atb::Tensor) = &SqueezeBatchSeq;
            if (CreateTensor(variantPack.outTensors.at(i), i, aclnnVariantPack.aclOutTensors[i],
                             ReshapeTensorDecsFuncPtr) != atb::NO_ERROR) {
                LOG_ERROR_MODEL << this->opName_ << " OutTensor aclCreateTensor index " << i << " fail";
                return atb::ERROR_INTERNAL_ERROR;
            }
            LOG_DEBUG_MODEL << opName_ << " CreateTensor end";
        }
        return atb::NO_ERROR;
    }

    int GeluOperation::SetAclNNWorkspaceExecutor()
    {
        LOG_DEBUG_MODEL << opName_ << " SetAclNNWorkspaceExecutor start, geluApproximate: " << param_.geluApproximate;
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        if (param_.geluApproximate == -1) {
            int ret = aclnnGeluGetWorkspaceSize(
                aclnnVariantPack.aclInTensors.at(0)->tensor,   // self
                aclnnVariantPack.aclOutTensors.at(0)->tensor,  // out
                &this->aclnnOpCache_->workspaceSize,
                &this->aclnnOpCache_->aclExecutor);
            LOG_DEBUG_MODEL << opName_ << " SetAclNNWorkspaceExecutor end"
                        << ", ret: " << ret
                        << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                        << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor;
            return ret;
        } else {
            int ret = aclnnGeluV2GetWorkspaceSize(
                aclnnVariantPack.aclInTensors.at(0)->tensor,   // x
                param_.geluApproximate,                        // approximate
                aclnnVariantPack.aclOutTensors.at(0)->tensor,  // y
                &this->aclnnOpCache_->workspaceSize,
                &this->aclnnOpCache_->aclExecutor);
            LOG_DEBUG_MODEL << opName_ << " SetAclNNWorkspaceExecutor end"
                        << ", ret: " << ret
                        << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                        << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor;
            return ret;
        }
    }

    int GeluOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        LOG_DEBUG_MODEL << opName_ << " ExecuteAclNNOp start";
        if (param_.geluApproximate == -1) {
            int ret = aclnnGelu(
                workspace,
                this->aclnnOpCache_->workspaceSize,
                this->aclnnOpCache_->aclExecutor,
                stream);
            LOG_DEBUG_MODEL << opName_ << " ExecuteAclNNOp end, ret: " << ret;
            return ret;
        } else {
            int ret = aclnnGeluV2(
                workspace,
                this->aclnnOpCache_->workspaceSize,
                this->aclnnOpCache_->aclExecutor,
                stream);
            LOG_DEBUG_MODEL << opName_ << " ExecuteAclNNOp end, ret: " << ret;
            return ret;
        }
    }

}  // namespace atb_speed::common
