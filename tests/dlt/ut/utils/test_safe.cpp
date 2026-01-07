/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include <iostream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "safe_io.h"

namespace mindie_llm {

constexpr int TestJsonDepth = 10;

class TestSafe : public ::testing::Test {
void SetUp() override {
    }
};

std::string TestCreatNestedJson(int depth)
{
    std::string out;
    for (int i = 0; i < depth; i++) {
        out += "{\"a\":";
    }
    out += "b";
    for (int i = 0; i < depth; i++) {
        out += "}";
    }
    return out;
}

std::string TestCreatNestedArrayJson(int depth)
{
    std::string out;
    for (int i = 0; i < depth; i++) {
        out += "[";
    }
    out += "b";
    for (int i = 0; i < depth; i++) {
        out += "]";
    }
    return out;
}

int TestGetJsonDepth(const nlohmann::json& j)
{
    if (j.is_object()) {
        int maxLevel = 0;
        for (auto& element : j.items()) {
            if (element.value().is_object()) {
                int subLevel = TestGetJsonDepth(element.value()) + 1;
                if (subLevel > maxLevel) {
                    maxLevel = subLevel;
                }
            }
        }
        return maxLevel;
    } else if (j.is_array()) {
        int maxLevel = 0;
        for (auto& element : j) {
            if (element.is_object()) {
                int subLevel = TestGetJsonDepth(element) + 1;
                if (subLevel > maxLevel) {
                    maxLevel = subLevel;
                }
            }
        }
        return maxLevel;
    } else {
        // 如果不是对象或数组，返回0（基本类型不增加嵌套层次）
        return 0;
    }
}

TEST_F(TestSafe, TestCheckJsonDepthCallback)
{
    nlohmann::json dummObj{};
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth, Json::parse_event_t::object_start, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth, Json::parse_event_t::array_start, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth, Json::parse_event_t::object_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth, Json::parse_event_t::array_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth, Json::parse_event_t::key, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth, Json::parse_event_t::valu, dummObj));
    ASSERT_EQ(false, CheckJsonDepthCallbackNoLogger(TestJsonDepth + 1, Json::parse_event_t::object_start, dummObj));
    ASSERT_EQ(false, CheckJsonDepthCallbackNoLogger(TestJsonDepth + 1, Json::parse_event_t::array_start, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth + 1, Json::parse_event_t::object_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth + 1, Json::parse_event_t::array_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth + 1, Json::parse_event_t::key, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackNoLogger(TestJsonDepth + 1, Json::parse_event_t::valu, dummObj));

    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth, Json::parse_event_t::object_start, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth, Json::parse_event_t::array_start, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth, Json::parse_event_t::object_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth, Json::parse_event_t::array_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth, Json::parse_event_t::key, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth, Json::parse_event_t::valu, dummObj));
    ASSERT_EQ(false, CheckJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::object_start, dummObj));
    ASSERT_EQ(false, CheckJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::array_start, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::object_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::array_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::key, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::valu, dummObj));

    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth, Json::parse_event_t::object_start, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth, Json::parse_event_t::array_start, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth, Json::parse_event_t::object_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth, Json::parse_event_t::array_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth, Json::parse_event_t::key, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth, Json::parse_event_t::valu, dummObj));
    ASSERT_EQ(false, CheckJsonDepthCallbackUlog(TestJsonDepth + 1, Json::parse_event_t::object_start, dummObj));
    ASSERT_EQ(false, CheckJsonDepthCallbackUlog(TestJsonDepth + 1, Json::parse_event_t::array_start, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth + 1, Json::parse_event_t::object_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth + 1, Json::parse_event_t::array_end, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth + 1, Json::parse_event_t::key, dummObj));
    ASSERT_EQ(true, CheckJsonDepthCallbackUlog(TestJsonDepth + 1, Json::parse_event_t::valu, dummObj));

    nlohmann::ordered_json dummObj2{};
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth, Json::parse_event_t::object_start, dummObj));
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth, Json::parse_event_t::array_start, dummObj));
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth, Json::parse_event_t::object_end, dummObj));
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth, Json::parse_event_t::array_end, dummObj));
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth, Json::parse_event_t::key, dummObj));
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth, Json::parse_event_t::valu, dummObj));
    ASSERT_EQ(false, CheckOrderedJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::object_start, dummObj));
    ASSERT_EQ(false, CheckOrderedJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::array_start, dummObj));
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::object_end, dummObj));
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::array_end, dummObj));
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::key, dummObj));
    ASSERT_EQ(true, CheckOrderedJsonDepthCallback(TestJsonDepth + 1, Json::parse_event_t::valu, dummObj));
}

TEST_F(TestSafe, TestCheckJsonDepthCallbackWithParse)
{
    std::string strNest10 = TestCreatNestedJson(TestJsonDepth);
    auto objBase = nlohmann::json::parse(strNest10);
    int depBase = TestGetJsonDepth(objBase);
    ASSERT_EQ(depBase, TestJsonDepth);

    auto objNest10 = nlohmann::json::parse(strNest10, CheckJsonDepthCallbackNoLogger);
    int depNest10 = TestGetJsonDepth(objNest10);
    ASSERT_EQ(depNest10, TestJsonDepth);

    std::string strNest11 = TestCreatNestedJson(TestJsonDepth + 1);
    auto objNest11 = nlohmann::json::parse(strNest11, CheckJsonDepthCallbackNoLogger);
    int depNest11 = TestGetJsonDepth(objNest11);
    ASSERT_EQ(depNest11, TestJsonDepth + 1);

    std::string strArr10 = TestCreatNestedArrayJson(TestJsonDepth);
    auto arrBase = nlohmann::json::parse(strArr10);
    depBase = TestGetJsonDepth(arrBase);
    ASSERT_EQ(depBase, TestJsonDepth);

    auto arrNest10 = nlohmann::json::parse(strArr10, CheckJsonDepthCallbackNoLogger);
    int depArr10 = TestGetJsonDepth(arrNest10);
    ASSERT_EQ(depArr10, TestJsonDepth);

    std::string strArr11 = TestCreatNestedArrayJson(TestJsonDepth + 1);
    auto arrNest11 = nlohmann::json::parse(strArr11, CheckJsonDepthCallbackNoLogger);
    int depArr11 = TestGetJsonDepth(arrNest11);
    ASSERT_EQ(depArr11, TestJsonDepth + 1);
}

} // namespace mindie_llm