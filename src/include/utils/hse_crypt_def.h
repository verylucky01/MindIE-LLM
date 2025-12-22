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
 
#ifndef HSECEASY_HSE_CRYPT_COMMON_H
#define HSECEASY_HSE_CRYPT_COMMON_H

#include <string>
#include <vector>

namespace ock {
namespace hse {
using CResult = int32_t;

using CryptCode = enum CCode : CResult {
    HSE_CRYPT_OK = 0,                          // Success
    HSE_CRYPT_ERROR = -1,                      // Failed
    HSE_CRYPT_PARAM_INVALID = -2,              // The param is invalid
    HSE_CRYPT_PARAM_DOMAIN_COUNT_INVALID = -3, // the domain count is invalid
    HSE_CRYPT_KMC_INIT_FAILED = -4,            // Init kmc failed
    HSE_CRYPT_KEY_ACTIVE_FAILED = -5           // Active kmc key failed
};

constexpr int32_t ERROR_CODE_STRING_COUNT = 6;
const static std::vector<std::string> g_errorCodeString = {
    "No error",
    "Inner error",
    "Parameter is invalid",
    "Domain count is invalid",
    "KMC init failed",
    "KMC key active failed"
};

enum class CryptLogLevel {
    HSE_CRYPT_LOG_DISABLE = 0,
    HSE_CRYPT_LOG_ERROR,
    HSE_CRYPT_LOG_WARN,
    HSE_CRYPT_LOG_INFO,
    HSE_CRYPT_LOG_DEBUG,
    HSE_CRYPT_LOG_TRACE
};

enum class EncryptAlg {
    HSE_CRYPT_AES128_GCM,
    HSE_CRYPT_AES256_GCM,
    HSE_CRYPT_SM4_CTR,
};

enum class HmacAlg {
    HSE_CRYPT_HMAC_SHA256,
    HSE_CRYPT_HMAC_SHA384,
    HSE_CRYPT_HMAC_SHA512,
};

}
}

#endif // HSECEASY_HSE_CRYPT_COMMON_H

