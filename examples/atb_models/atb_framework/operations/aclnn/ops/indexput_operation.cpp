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
#include "indexput_operation.h"
#include "acl/acl.h"
#include "system_log.h"
#include "aclnnop/aclnn_index_put_impl.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {
IndexputOperation::IndexputOperation(const std::string &name, AclNNIndexputParam param)
    : AclNNOperation(name), param_(param)
{
    LOG_DEBUG_MODEL << "IndexputOperation, param: " << param_.ToString();
}

IndexputOperation::~IndexputOperation()
{
    LOG_DEBUG_MODEL << "~IndexputOperation";
    this->DestroyOperation();
}

uint32_t IndexputOperation::GetInputNum() const { return NUM3; }

uint32_t IndexputOperation::GetOutputNum() const { return NUM1; }

atb::Status IndexputOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                          atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    LOG_DEBUG_MODEL << opName_ << " infer shape start";
    outTensorDescs.at(0) = inTensorDescs.at(0);
    LOG_DEBUG_MODEL << opName_ << " infer shape end";
    return atb::NO_ERROR;
}

atb::Status IndexputOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    LOG_DEBUG_MODEL << opName_ << " CreateAclNNVariantPack start";
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    uint32_t inputNum = this->GetInputNum();
    aclnnVariantPack.aclInTensors.resize(inputNum);
    for (uint32_t i = 0; i < inputNum; ++i) {
        CHECK_OPERATION_STATUS_RETURN(
            CreateTensor(variantPack.inTensors.at(i), static_cast<size_t>(i), aclnnVariantPack.aclInTensors[i]));
        if (i == 1) {
            aclnnVariantPack.aclInTensors.at(i)->tensorListidx = 0;
            aclnnVariantPack.aclInTensors.at(i)->tensorIdx = 0;
        }
    }

    vectorList.clear();
    vectorList.push_back(aclnnVariantPack.aclInTensors.at(1)->tensor);
    aclnnVariantPack.aclInTensorList.clear();
    aclnnVariantPack.aclInTensorList.push_back(aclCreateTensorList(vectorList.data(), vectorList.size()));

    return atb::NO_ERROR;
}

atb::Status IndexputOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.clear();
    CHECK_OPERATION_STATUS_RETURN(CreateTensor(variantPack.outTensors.at(0), 0, aclnnVariantPack.aclOutTensors[0]));
    LOG_DEBUG_MODEL << opName_ << " CreateAclNNVariantPack end";
    return atb::NO_ERROR;
}

int IndexputOperation::SetAclNNWorkspaceExecutor()
{
    LOG_DEBUG_MODEL << opName_ << " SetAclNNWorkspaceExecutor start";
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    int ret = aclnnIndexPutImplGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor, aclnnVariantPack.aclInTensorList.at(0),
        aclnnVariantPack.aclInTensors.at(2)->tensor, param_.accumulate, param_.unsafe,
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    LOG_DEBUG_MODEL << opName_ << " end, ret:" << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor;
    LOG_DEBUG_MODEL << opName_ << " SetAclNNWorkspaceExecutor end";
    return ret;
}

int IndexputOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    LOG_DEBUG_MODEL << opName_ << " ExecuteAclNNOp start";
    int ret =
        aclnnIndexPutImpl(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    LOG_DEBUG_MODEL << opName_ << " ExecuteAclNNOp end, ret:" << ret;
    return ret;
}

} // namespace common
} // namespace atb_speed