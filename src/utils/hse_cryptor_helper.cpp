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
 
#include <memory>
#include <fstream>
#include <sys/ipc.h>

#include "hse_cryptor.h"
#include "env_util.h"
#include "log.h"
#include "file_utils.h"
#include "hse_cryptor_helper.h"

using namespace ock::hse;
using namespace mindie_llm;
std::mutex HseCryptorHelper::globalMutex;

static constexpr int MIN_VALID_KEY = 0x20161111;
static constexpr int MAX_VALID_KEY = 0x20169999;

HseCryptorHelper::HseCryptorHelper(std::string kfsMaster, std::string kfsStandby) noexcept
    : kfsMasterPath{ std::move(kfsMaster) }, kfsStandbyPath{ std::move(kfsStandby) }
{}

int HseCryptorHelper::Decrypt(int domainId, const std::string &filePath, const std::string &baseDir,
    std::pair<char *, int> &result) noexcept
{
    std::string encryptedText;
    auto ret = ReadFile(filePath, baseDir, encryptedText);
    if (ret != 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SECURE,
            ABNORMAL_TRANSMISSION_ERROR), "Read private key file failed.");
        return -1;
    }
    std::unique_ptr<char[]> bufferPtr = std::make_unique<char[]>(encryptedText.length());
    if (!bufferPtr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SECURE,
            ABNORMAL_TRANSMISSION_ERROR), "Allocate memory for buffer failed.");
        return -1;
    }

    HelperInitializer initializer{ kfsMasterPath, kfsStandbyPath };
    if (!initializer.Initialized()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SECURE,
            ABNORMAL_TRANSMISSION_ERROR), "Initialize key file ksfa/ksfb failed.");
        return -1;
    }

    ret = HseCryptor::RefreshMkMask();
    if (ret != 0) {
        std::string logMsg = "Refresh mk mask failed, ret code is " + std::to_string(ret);
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SECURE,
            ABNORMAL_TRANSMISSION_ERROR), logMsg.c_str());
        return -1;
    }

    auto dataLength = static_cast<int>(encryptedText.length());
    ret = HseCryptor::Decrypt(domainId, encryptedText, bufferPtr.get(), dataLength);
    if (ret != 0) {
        std::string logMsg = "Decrypt tls key password failed, return code is " + std::to_string(ret);
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SECURE,
            ABNORMAL_TRANSMISSION_ERROR), logMsg.c_str());
        return -1;
    }

    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Decrypt success.");
    result = std::make_pair(bufferPtr.release(), dataLength);
    return 0;
}

int HseCryptorHelper::CheckMasterKeyExpired(int domainId, bool &isRkExpired, uint32_t lead) noexcept
{
    HelperInitializer initializer{ kfsMasterPath, kfsStandbyPath };
    if (!initializer.Initialized()) {
        return -1;
    }
    std::time_t expiredTimeStamp = std::time(nullptr);
    return HseCryptor::CheckMasterKeyExpired(domainId, isRkExpired, lead, expiredTimeStamp);
}

void HseCryptorHelper::EraseDecryptData(std::pair<char *, int> &data)
{
    if (data.first != nullptr) {
        for (auto i = 0; i < data.second; i++) {
            data.first[i] = '\0';
        }
        delete[] data.first;
        data.first = nullptr;
    }
    data.second = 0;
}

int HseCryptorHelper::ReadFile(const std::string &path, const std::string &baseDir, std::string &content) noexcept
{
    bool checkFlag = true;
    const std::string isCheck = EnvUtil::GetInstance().Get("MINDIE_CHECK_INPUTFILES_PERMISSION");
    if (isCheck == "0") {
        checkFlag = false;
    }
    std::string errMsg;
    std::string regularPath;
    if (!FileUtils::RegularFilePath(path, baseDir, errMsg, regularPath) ||
        !FileUtils::IsFileValid(regularPath, errMsg, true, FileUtils::FILE_MODE_400, checkFlag)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SECURE,
            ABNORMAL_TRANSMISSION_ERROR), errMsg);
        return -1;
    }
    std::ifstream in(regularPath);
    if (!in.is_open()) {
        std::string safePath = FileUtils::GetSafeRelativePath(regularPath);
        std::string logMsg = "Failed to open file " + safePath;
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SECURE,
            ABNORMAL_TRANSMISSION_ERROR), logMsg.c_str());
        return -1;
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    content = buffer.str();
    in.close();
    return 0;
}

HseCryptorHelper::HelperInitializer::HelperInitializer(const std::string &m, const std::string &s) noexcept
    : initialized{ false }, lockGuard{ globalMutex }
{
    KmcCryptConfigBuilder builder;
    auto key = ftok(m.c_str(), 0);
    if (key < 0) {
        std::string logMsg = "Get key failed: " + std::to_string(errno) + ":" + strerror(errno);
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SECURE,
            ABNORMAL_TRANSMISSION_ERROR), logMsg.c_str());
        return;
    }

    auto config = builder.MasterKsfFile(m)
                      .StandByKsfFile(s)
                      .SemKey(MIN_VALID_KEY + key % (MAX_VALID_KEY - MIN_VALID_KEY))
                      .LogLevel(CryptLogLevel::HSE_CRYPT_LOG_INFO)
                      .Build();
    auto ret = HseCryptor::Initialize(config);
    if (ret != 0) {
        std::string logMsg = "Cryptor init failed: " + std::to_string(ret);
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SECURE,
            ABNORMAL_TRANSMISSION_ERROR), logMsg.c_str());
        return;
    }

    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Cryptor initialize success.");
    initialized = true;
}

HseCryptorHelper::HelperInitializer::~HelperInitializer() noexcept
{
    if (initialized) {
        HseCryptor::UnInitialize();
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Cryptor un-initialized.");
    }
}

bool HseCryptorHelper::HelperInitializer::Initialized() const noexcept
{
    return initialized;
}
