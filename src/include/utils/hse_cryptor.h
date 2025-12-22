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

#ifndef HSE_CRYPTOR_FACTORY_H
#define HSE_CRYPTOR_FACTORY_H

#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <string>
#include <climits>
#include <sys/stat.h>
#include "hse_crypt_def.h"
#include "file_utils.h"

namespace ock {
namespace hse {
class HseCryptConfig {
public:
    std::string masterKsfFilePath;
    std::string standKsfFilePath;
    int domainCount = 2;
    int semKey = 0x20161316; //  0x20161316 default sem key
    EncryptAlg encryptAlg = EncryptAlg::HSE_CRYPT_AES256_GCM;
    HmacAlg hmacAlg = HmacAlg::HSE_CRYPT_HMAC_SHA512;
    CryptLogLevel logLevel = CryptLogLevel::HSE_CRYPT_LOG_INFO;
};


/// The KmcCryptConfigBuilder class is used to configure the KMC.
///
/// The KmcCryptConfigBuilder class provides methods to configure the ksf file, domain count, log
/// level, encrypt algorithm, hmac algorithm and sem key.
class KmcCryptConfigBuilder {
public:
    /// Create KmcCryptConfig
    ///
    /// \return The instance of KmcCryptConfig
    HseCryptConfig Build()
    {
        return config;
    }

    /// Set the path of masterKeyFile
    ///
    /// \param masterKeyFile The path of master key file.
    /// \return The KmcCryptConfigBuilder.
    KmcCryptConfigBuilder MasterKsfFile(std::string masterKeyFile)
    {
        config.masterKsfFilePath = masterKeyFile;
        return *this;
    }

    /// Set the path of standbyKeyFile.
    ///
    /// \param standbyKeyFile The path of standby key file.
    /// \return KmcCryptConfigBuilder
    KmcCryptConfigBuilder StandByKsfFile(std::string standbyKeyFile)
    {
        config.standKsfFilePath = standbyKeyFile;
        return *this;
    }

    /// Set domain count.
    ///
    /// \param domainCount The domain number.
    /// \return The KmcCryptConfigBuilder.
    KmcCryptConfigBuilder DomainCount(int domainCount)
    {
        config.domainCount = domainCount;
        return *this;
    }

    /// Set log level.
    ///
    /// \param logLevel see@CryptLogLevel.
    /// \return The KmcCryptConfigBuilder.
    KmcCryptConfigBuilder LogLevel(CryptLogLevel logLevel)
    {
        config.logLevel = logLevel;
        return *this;
    }

    /// Set encrypt algorithm.
    ///
    /// \param encryptAlg  see@EncryptAlg.
    /// \return The KmcCryptConfigBuilder.
    KmcCryptConfigBuilder EncryptAlgorithm(EncryptAlg encryptAlg)
    {
        config.encryptAlg = encryptAlg;
        return *this;
    }

    /// Set hmac algorithm.
    ///
    /// \param hmacAlg see@HmacAlg.
    /// \return The KmcCryptConfigBuilder.
    KmcCryptConfigBuilder HmacAlgorithm(HmacAlg hmacAlg)
    {
        config.hmacAlg = hmacAlg;
        return *this;
    }

    /// Set semantic key.
    ///
    /// \param semKey The semantic key.
    /// \return The KmcCryptConfigBuilder.
    KmcCryptConfigBuilder SemKey(int semKey)
    {
        config.semKey = semKey;
        return *this;
    }

private:
    HseCryptConfig config;
};

#ifndef HSE_EXTERNAL_LOG_FUNC
using SecEasyExternalLog = void (*)(int level, const char *msg);
#endif

class HseCryptor;

using HSE_CRYPTION_CREATE_FUNCTION = int32_t (*)(HseCryptor **);
static std::string g_hseCryptLibraryName("libhse_cryption.so");
static HSE_CRYPTION_CREATE_FUNCTION g_hseCryptorCreator = nullptr;

static int GetLibPath(const std::string &fileName, std::string &realPath)
{
    std::string envHseceasyHome = (getenv("HSECEASY_PATH") == nullptr) ? "" : getenv("HSECEASY_PATH");
    if (envHseceasyHome.empty()) {
        std::cout << "Environment variable \"HSECEASY_PATH\" is not set. Exit." << std::endl;
        return -1;
    }
    if (envHseceasyHome.back() != '/') {
        envHseceasyHome += "/";
    }
    std::string path = envHseceasyHome + fileName;
    if (path.length() > PATH_MAX) {
        return -1;
    }

    struct stat buf;
    if (lstat(path.c_str(), &buf) != 0) {
        return -1;
    }
    if (S_ISLNK(buf.st_mode)) {
        return -1;
    }

    char pathBuf[PATH_MAX + 1] = {0};
    char *realPathChar = realpath(path.c_str(), pathBuf);
    if (realPathChar == nullptr) {
        std::cout << "Failed to get real path:" << path << std::endl;
        return -1;
    }
    realPath = realPathChar;
    return 0;
}

static int32_t HseLoadCryptLibrary()
{
    if (g_hseCryptorCreator != nullptr) {
        return HSE_CRYPT_OK;
    }
    std::string realPath;
    auto ret = GetLibPath(g_hseCryptLibraryName, realPath);
    if (ret != 0) {
        std::cout << "Failed to get path for '" << g_hseCryptLibraryName << "'" << std::endl;
        return -1;
    }

    std::string errMsg;
    if (!mindie_llm::FileUtils::IsFileValid(realPath, errMsg, true, mindie_llm::FileUtils::FILE_MODE_640, true)) {
        std::cout << "Failed to check file valid and permission 640 " << g_hseCryptLibraryName << ", error:"
                    <<
                    std::endl;
        return -1;
    }

    auto handle = dlopen(realPath.c_str(), RTLD_NOW | RTLD_DEEPBIND);
    if (handle == nullptr) {
        std::cout << "Failed to load the library " << g_hseCryptLibraryName << ", error:" << dlerror() << "."
                    <<
                    std::endl;
        return HSE_CRYPT_ERROR;
    }

    g_hseCryptorCreator = (HSE_CRYPTION_CREATE_FUNCTION) dlsym(handle, "GetHseCryptoCreateInstance");
    if (g_hseCryptorCreator == nullptr) {
        std::cout << "Failed to call dlsym to load function , error:" << dlerror() << "." << std::endl;
        dlclose(handle);
        return HSE_CRYPT_ERROR;
    }
    return HSE_CRYPT_OK;
}


/// The HseCryptor class is used to manage the key and certificate.
///
/// The HseCryptor class provides methods to check and update the master key or root key, and
/// encrypt or decrypt specific content etc.
class HseCryptor {
public:
    /// Initialize the component of HseCryptor.
    ///
    /// \param config see@HseCryptConfig.
    /// \return HSE_CRYPT_OK success, otherwise failed.
    static int32_t Initialize(HseCryptConfig &config)
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoInitialize(config);
    }

    /// UnInitialize the component.
    ///
    /// \return HSE_CRYPT_OK success, otherwise failed.
    static int32_t UnInitialize()
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoUnInitialize();
    }

    /// Check whether root key expired.
    ///
    /// \param isRkExpired [out] True if expired, otherwise unexpired.
    /// \param lead The days ahead certificate expires.
    /// \param rootKeyExpiredTime:set the time for key expiration notification.
    /// \return HSE_CRYPT_OK if success, otherwise failed.
    static int32_t CheckRootKeyExpired(bool &isRkExpired, uint32_t lead, time_t &rootKeyExpiredTime)
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoCheckRootKeyExpired(isRkExpired, lead, &rootKeyExpiredTime);
    }

    /// Update root key if expired.
    ///
    /// \param lead The days ahead certificate expires.
    /// \return HSE_CRYPT_OK if success, otherwise failed.
    static int32_t CheckRootKeyExpiredAndUpdate(uint32_t lead)
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoCheckRootKeyAndAutoUpdate(lead);
    }

    /// Encrypt a plain text into cipher text.
    /// \param domainId           [in] Domain id of kmc for encryption the value
    ///                                between 0~domainCount see@HseCryptConfig.
    /// \param plainText          [in] Plain text need to be encrypted.
    /// \param plainTextLen       [in] Length of plain text need to be encrypted.
    /// \param cipherText         [out] Encrypted string.
    /// \return HSE_CRYPT_OK if success, otherwise failed.
    static int32_t
    Encrypt(int32_t domainId, const char *plainText, uint32_t plainTextLen, std::string &cipherText)
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoEncrypt(domainId, plainText, plainTextLen, cipherText);
    }

    /// Decrypt a cipher text into plain text.
    ///
    /// \param domainId         [in] Domain id of kmc for encryption. the value
    ///                              between 0~domainCount see@HseCryptConfig.
    /// \param cipherText       [in] Encrypted string.
    /// \param plainText        [out] plain text need to be encrypted, the buffer
    ///                               length at least equals the length of cipherText.
    /// \param plainTextLen   [in, out] length of plain text need to be encrypted.
    /// \return HSE_CRYPT_OK if success, otherwise failed.
    static int32_t
    Decrypt(int32_t domainId, const std::string &cipherText, char *plainText, int32_t &plainTextLen)
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoDecrypt(domainId, cipherText, plainText, plainTextLen);
    }

    /// Set the logger.
    ///
    /// \param f The logger callback.
    /// \return HSE_CRYPT_OK if success, otherwise failed.
    static int32_t SetExternalLogger(SecEasyExternalLog f)
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoSetExternalLogger(f);
    }

    /// Get the description of error code.
    ///
    /// \param errorCode.
    /// \return The description of error code.
    static const std::string GetErrStr(int32_t errorCode)
    {
        auto tmpCode = 0 - errorCode;
        if (tmpCode < 0 || tmpCode >= ERROR_CODE_STRING_COUNT) {
            return "";
        }
        return g_errorCodeString[tmpCode];
    }

    /// Update master key if expired.
    ///
    /// \param domainId The domain of the master key.
    /// \param lead The days ahead certificate expires.
    /// \return HSE_CRYPT_OK if success, otherwise failed.
    static int32_t CheckMasterKeyExpiredAndAutoUpdate(int32_t domainId, int32_t lead)
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoCheckMasterKeyExpiredAndAutoUpdate(domainId, lead);
    }

    /// Check if master key has expired.
    ///
    /// \param domainId the domain of the master key.
    /// \param isRkExpired  [out] True if expired, otherwise unexpired.
    /// \param lead The days ahead certificate expires.
    /// \return HSE_CRYPT_OK if success, otherwise failed.
    static int32_t
    CheckMasterKeyExpired(int domainId, bool &isRkExpired, uint32_t lead, time_t &expiredTimeStamp)
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoCheckMasterKeyExpired(domainId, isRkExpired, lead, &expiredTimeStamp);
    }

    /// Refresh the key mask.
    ///
    /// It's suggested to call this interface periodically every two hour.
    ///
    /// \return HSE_CRYPT_OK if success, otherwise failed.
    static int32_t RefreshMkMask()
    {
        HseCryptor *hseCryptor = GetInstance();
        if (hseCryptor == nullptr) {
            return HSE_CRYPT_ERROR;
        }
        return hseCryptor->DoRefreshMkMask();
    }

protected:
    virtual int32_t DoInitialize(const HseCryptConfig &config) = 0;

    virtual int32_t DoUnInitialize() = 0;

    virtual int32_t DoCheckRootKeyExpired(bool &isRkExpired, uint32_t lead, time_t *rootKeyExpiredTime) = 0;

    virtual int32_t DoSetExternalLogger(SecEasyExternalLog f) = 0;

    virtual int32_t DoEncrypt(int32_t domainId, const char *plainText, uint32_t plainTextLen,
                                std::string &cipherText) = 0;

    virtual int32_t DoDecrypt(int32_t domainId, const std::string &cipherText, char *&plainText,
                                int32_t &plainTextLen) = 0;

    virtual int32_t DoCheckMasterKeyExpiredAndAutoUpdate(int32_t domainId, uint32_t lead) = 0;

    virtual int32_t DoCheckRootKeyAndAutoUpdate(uint32_t lead) = 0;

    virtual int32_t
    DoCheckMasterKeyExpired(int32_t domainId, bool &isRkExpired, uint32_t lead, time_t *expiredTimeStamp) = 0;

    virtual int32_t DoRefreshMkMask() = 0;

private:
    static HseCryptor *GetInstance()
    {
        static HseCryptor *instance = nullptr;
        static std::mutex instanceLock;
        if (instance == nullptr) {
            std::lock_guard <std::mutex> guard(instanceLock);
            if (instance == nullptr) {
                int result = HSE_CRYPT_OK;
                HseCryptor *cryptor = nullptr;
                if ((result = HseLoadCryptLibrary()) != HSE_CRYPT_OK) {
                    return nullptr;
                }
                if ((result = g_hseCryptorCreator(&cryptor)) != 0) {
                    std::cout << "Get cryptor instance failed." << std::endl;
                    return nullptr;
                }
                instance = cryptor;
            }
        }
        return instance;
    }
};
}
}
#endif // HSE_CRYPTOR_FACTORY_H

