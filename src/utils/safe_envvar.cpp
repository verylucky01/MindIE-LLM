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
#include "safe_envvar.h"

namespace mindie_llm {

EnvVar& EnvVar::GetInstance()
{
    static EnvVar instance;
    return instance;
}

Result EnvVar::Set(const char *key, const std::string& value, bool overwrite) const
{
    if (!key || value.empty()) {
        return Result::Error(ResultCode::NONE_ARGUMENT,
            "Environment variable key is null or value is an empty string.");
    }
    int ret = setenv(key, value.c_str(), overwrite ? 1 : 0);
    if (ret != 0) {
        return Result::Error(
            ResultCode::IO_FAILURE,
            "Failed to set environment variable, errno: " + std::to_string(errno) + " for key: " + std::string(key)
        );
    }
    return Result::OK();
}

Result EnvVar::Get(const char *key, const std::string& defaultValue, std::string& outValue) const
{
    if (!key || defaultValue.empty()) {
        return Result::Error(ResultCode::NONE_ARGUMENT,
            "Environment variable key is nullptr or default value is empty.");
    }
    try {
        const char* val = std::getenv(key);
        outValue = (val) ? std::string(val) : defaultValue;
    } catch (const std::exception& e) {
        return Result::Error(ResultCode::IO_FAILURE, e.what());
    } catch (...) {
        return Result::Error(ResultCode::IO_FAILURE, "Unknown error occurred while fetching environment variable");
    }
    return Result::OK();
}

} // namespace mindie_llm
