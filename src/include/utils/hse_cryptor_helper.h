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

#ifndef ENDPOINT_HSECRYPTORHELPER_H
#define ENDPOINT_HSECRYPTORHELPER_H

#include <cstdint>
#include <string>
#include <mutex>

namespace mindie_llm {

constexpr int MIN_PRIVATE_KEY_CONTENT_BIT_LEN = 3072; // RSA密钥长度要求大于3072
constexpr int MAX_PRIVATE_KEY_CONTENT_BIT_LEN = 32768; // huge RSA秘钥位宽为32768

constexpr int MIN_PRIVATE_KEY_CONTENT_BYTE_LEN = MIN_PRIVATE_KEY_CONTENT_BIT_LEN / 8; // 1个byte = 8bit
constexpr int MAX_PRIVATE_KEY_CONTENT_BYTE_LEN = MAX_PRIVATE_KEY_CONTENT_BIT_LEN / 8; // 1个byte = 8bit

constexpr int MAX_TOKEN_LEN = 128; // 私钥口令上限是128字符
constexpr int MIN_TOKEN_LEN = 8; // 私钥口令下限是8字符

class HseCryptorHelper {
public:
    HseCryptorHelper(std::string kfsMaster, std::string kfsStandby) noexcept;

public:
    int Decrypt(int domainId, const std::string &filePath, const std::string &baseDir,
        std::pair<char *, int> &result) noexcept;
    int CheckMasterKeyExpired(int domainId, bool &isRkExpired, uint32_t lead) noexcept;
    static void EraseDecryptData(std::pair<char *, int> &data);
    static int ReadFile(const std::string &path, const std::string &baseDir, std::string &content) noexcept;

private:
    class HelperInitializer {
    public:
        HelperInitializer(const std::string &m, const std::string &s) noexcept;
        ~HelperInitializer() noexcept;

    public:
        bool Initialized() const noexcept;

    private:
        bool initialized;
        std::unique_lock<std::mutex> lockGuard;
    };

private:
    const std::string kfsMasterPath;
    const std::string kfsStandbyPath;
    static std::mutex globalMutex;
};
} // namespace mindie_llm

#endif // ENDPOINT_HSECRYPTORHELPER_H
