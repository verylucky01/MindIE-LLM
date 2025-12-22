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
#include "mockcpp/mockcpp.hpp"
#include "http_wrapper.h"
#include "http_server.h"

using namespace mindie_llm;

namespace mindie_llm {
class HttpWrapperTest : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }

    HttpWrapper httpWrapper;
};

TEST_F(HttpWrapperTest, Instance)
{
    auto gHttpWrapper = HttpWrapper::Instance();
    EXPECT_TRUE(gHttpWrapper != nullptr);
}

TEST_F(HttpWrapperTest, StartSuccess)
{
    MOCKER(HttpServer::HttpServerInit).stubs().will(returnValue(0));
    int32_t ret = httpWrapper.Start();
    EXPECT_EQ(ret, 0);
    httpWrapper.Stop();
}

TEST_F(HttpWrapperTest, StartFail)
{
    MOCKER(HttpServer::HttpServerInit).stubs().will(returnValue(1U));
    int32_t ret = httpWrapper.Start();
    EXPECT_EQ(ret, 1);
    httpWrapper.Stop();
}

} // namespace llm