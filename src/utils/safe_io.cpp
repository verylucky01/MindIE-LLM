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
#include "safe_io.h"

#include <fstream>

#include "safe_path.h"

namespace mindie_llm {

Result LoadJson(const std::string& path, Json& json)
{
    std::string checkedPath;
    SafePath inFile(path, PathType::FILE, "r", SIZE_500MB, ".json");
    Result r = inFile.Check(checkedPath);
    if (!r.IsOk()) {
        return r;
    }
    std::ifstream file(checkedPath);
    if (!file.is_open()) {
        return Result::Error(ResultCode::IO_FAILURE, "Failed to open file: " + checkedPath);
    }
    file >> json;
    return Result::OK();
}

} // namespace mindie_llm
