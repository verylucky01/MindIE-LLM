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

#ifndef MEM_POOL_H
#define MEM_POOL_H

#include <pybind11/embed.h>
#include <iostream>
#include <string>
#include "log.h"

namespace py = pybind11;

namespace mindie_llm {

#pragma GCC visibility push(default)
class MemPool {
public:
    explicit MemPool(std::shared_ptr<py::object> instance) : impl_(std::move(instance)) {}

    bool LookUp(const std::string &key)
    {
        py::gil_scoped_acquire acquire;
        bool res = impl_->attr("exists")(key).cast<bool>();
        if (res) {
            MINDIE_LLM_LOG_DEBUG("Look up key=" << key << " sucessfully !!!");
        } else {
            MINDIE_LLM_LOG_DEBUG("Look up key=" << key << " failed !!!");
        }
        return res;
    }

private:
    std::shared_ptr<py::object> impl_{};
};

using MemPoolSPtr = std::shared_ptr<MemPool>;
#pragma GCC visibility pop
}  // namespace mindie_llm
#endif
