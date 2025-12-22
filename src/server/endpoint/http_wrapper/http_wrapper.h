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

#ifndef OCK_ENDPOINT_HTTP_WRAPPER_H
#define OCK_ENDPOINT_HTTP_WRAPPER_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>

namespace mindie_llm {
    class HttpWrapper {
    public:
        HttpWrapper() = default;
        ~HttpWrapper() = default;
        HttpWrapper(const HttpWrapper&) = delete;
        HttpWrapper& operator=(const HttpWrapper&) = delete;

        static HttpWrapper* Instance()
        {
            if (gHttpWrapper == nullptr) {
                std::lock_guard<std::mutex> lock(gInitMutex);
                if (gHttpWrapper == nullptr) {
                    gHttpWrapper = new (std::nothrow) HttpWrapper();
                    if (gHttpWrapper == nullptr) {
                        std::cout << "Failed to create new http wrapper, probably out of memory\n";
                    }
                }
            }
            return gHttpWrapper;
        }

        int32_t Start();
        void Stop();

    private:
        static std::mutex gInitMutex;
        static HttpWrapper* gHttpWrapper;
        std::mutex mMutex;
        bool mStarted{false};
    };
} // namespace mindie_llm

#endif // OCK_ENDPOINT_HTTP_WRAPPER_H
