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
#include <gtest/gtest.h>
#include "old_infer_req_res/infer_request.h"
#include "old_infer_req_res/infer_request_impl.h"

using namespace mindie_llm;

class InferRequestTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        InferRequestId reqId{"0"};
        request_ = std::make_shared<InferRequest>(reqId);
    }

    std::shared_ptr<InferRequest> request_;
};

TEST_F(InferRequestTest, TestPdRoleFlexPPercentage)
{
    request_->SetReqType(InferReqType::REQ_FLEX_LOCAL);
    bool flag = request_->IsFlexLocalReq();
    EXPECT_TRUE(flag);
}

class InferRequestImplTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        InferRequestId reqId{"0"};
        request_ = std::make_shared<InferRequestImpl>(reqId);
    }

    std::shared_ptr<InferRequestImpl> request_;
};

TEST_F(InferRequestImplTest, TestPdRoleFlexPPercentage)
{
    request_->SetReqType(InferReqType::REQ_FLEX_LOCAL);
    bool flag = request_->IsFlexLocalReq();
    EXPECT_TRUE(flag);
}