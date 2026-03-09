/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "index_select_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "system_log.h"
#include "aclnnop/aclnn_index_select.h"

namespace atb_speed::common {
    IndexSelectOperation::IndexSelectOperation(
        const std::string &name,
        atb_speed::common::IndexSelectParam param
    ) : AclNNOperation(name), param_(param)
    {
        LOG_DEBUG_MODEL << "IndexSelectOperation construct";
        this->opName_ = name;
    }

    IndexSelectOperation::~IndexSelectOperation()
    {
        LOG_DEBUG_MODEL << "IndexSelectOperation deconstruct";
        this->DestroyOperation();
    }

    uint32_t IndexSelectOperation::GetInputNum() const
    {
        return NUM2;  // inputTensorNum = 2
    }

    uint32_t IndexSelectOperation::GetOutputNum() const
    {
        return NUM1;  // outputTensorNum = 1
    }

    /**
     *
     * @param[in] inTensorDescs: [self, indices]
     * @param[in] outTensorDescs: out
     * @return atb::Status
     */
    atb::Status IndexSelectOperation::InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const
    {
        LOG_DEBUG_MODEL << opName_ << " InferShape start";
        outTensorDescs.at(0) = inTensorDescs.at(0);

        if (inTensorDescs.at(0).shape.dimNum > 8) {  // 8: tensor max dim num
            LOG_ERROR_MODEL << opName_ << " [input0 dimNum should <= 8] CHECK input0 dimNum = "
                            << inTensorDescs.at(0).shape.dimNum;
        }

        int64_t selfDimNum = static_cast<int64_t>(inTensorDescs.at(0).shape.dimNum);
        if ((param_.dim >= selfDimNum) || (param_.dim < -selfDimNum)) {
            LOG_ERROR_MODEL << opName_ << " [param dim should in [-input0 dimNum, input0 dimNum)) "
                            << "CHECK param dim = " << param_.dim << ", input0 dimNum = " << selfDimNum;
        }

        if (inTensorDescs.at(1).shape.dimNum != DIM1) {
            LOG_ERROR_MODEL << opName_ << " [input1 dimNum should == 1] CHECK input1 dimNum = "
                            << inTensorDescs.at(0).shape.dimNum;
        }
        outTensorDescs.at(0).shape.dims[param_.dim] = inTensorDescs.at(1).shape.dims[DIM0];
        LOG_DEBUG_MODEL << opName_ << " InferShape end";
        return atb::NO_ERROR;
    }

    int IndexSelectOperation::SetAclNNWorkspaceExecutor()
    {
        LOG_DEBUG_MODEL << opName_ << " SetAclNNWorkspaceExecutor start";
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        int ret = aclnnIndexSelectGetWorkspaceSize(
            aclnnVariantPack.aclInTensors.at(0)->tensor,     // self
            param_.dim,                                      // dim
            aclnnVariantPack.aclInTensors.at(1)->tensor,     // index
            aclnnVariantPack.aclOutTensors.at(0)->tensor,    // out
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        LOG_DEBUG_MODEL << opName_ << " SetAclNNWorkspaceExecutor end"
                      << ", ret: " << ret
                      << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                      << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor;
        return ret;
    }

    int IndexSelectOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        LOG_DEBUG_MODEL << opName_ << " ExecuteAclNNOp start";
        int ret = aclnnIndexSelect(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        LOG_DEBUG_MODEL << opName_ << " ExecuteAclNNOp end" << ", ret: " << ret;
        return ret;
    }

}  // namespace atb_speed::common
