/**
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
 
#ifndef MINDIE_LLM_STRING_UTILS_H
#define MINDIE_LLM_STRING_UTILS_H

#include <string>
#include <vector>
#include <map>
#include <sstream>

#include "safe_result.h"


namespace mindie_llm {
void Split(const std::string& text, char delimiter, std::vector<std::string>& tokens);
void Split(const std::string& text, const std::string& delimiter, std::vector<std::string>& tokens);
void RemoveSpaces(std::string& text);
bool IsSuffix(const std::string& str, const std::string& suffix);

template<typename T>
Result Str2Int(const std::string& str, const std::string& tag, T& outValue)
{
    if (str.empty() || tag.empty()) {
        return Result::Error(ResultCode::NONE_ARGUMENT,
            "The input string and corresponding tag cannot be empty string");
    }
    try {
        size_t idx = 0;
        outValue = static_cast<T>(std::stoi(str, &idx));
        if (idx != str.length()) {
            std::string msg = "For " + tag + ", contains invalid characters after parsing the integer: "
            + str.substr(idx);
            return Result::Error(ResultCode::PARSE_FAILURE, msg);
        }
    } catch (const std::exception& e) {
        return Result::Error(ResultCode::PARSE_FAILURE, "For " + tag + ", " + e.what());
    }
    return Result::OK();
}

template <typename T>
std::string Join(const std::vector<T>& vec, const std::string& delimiter)
{
    std::stringstream result;
    for (size_t i = 0; i < vec.size(); ++i) {
        result << vec[i];
        if (i != vec.size() - 1) {
            result << delimiter;
        }
    }
    return result.str();
}

template <typename KeyType, typename ValueType>
std::string GetKeysFromMap(const std::map<KeyType, ValueType>& m, const std::string& delimiter)
{
    std::stringstream result;
    for (auto it = m.begin(); it != m.end(); ++it) {
        result << it->first;
        if (std::next(it) != m.end()) {
            result << delimiter;
        }
    }
    return result.str();
}

} // namespace mindie_llm

#endif // MINDIE_LLM_STRING_UTILS_H