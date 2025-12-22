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

#include <fcntl.h>
#include "spdlog/details/os.h"
#include "common_util.h"
#include "check_utils.h"
#include "file_system.h"
#include "log_utils.h"
#include "log_config.h"

namespace mindie_llm {

LogConfig::LogConfig(const LogConfig &config)
    : logToStdOut_(config.logToStdOut_),
      logToFile_(config.logToFile_),
      logVerbose_(config.logVerbose_),
      logLevel_(config.logLevel_),
      baseDir_(config.baseDir_),
      logFilePath_(config.logFilePath_),
      logFileSize_(config.logFileSize_),
      logFileCount_(config.logFileCount_)
{
}

int LogConfig::Init(LoggerType loggerType)
{
    InitLogToStdoutFlag(loggerType);
    InitLogToFileFlag(loggerType);
    InitLogLevel(loggerType);
    InitLogFilePath(loggerType);
    InitLogVerbose(loggerType);
    InitLogRotationParam(loggerType);
    return LOG_OK;
}

void LogConfig::InitLogToStdoutFlag(LoggerType loggerType)
{
    const char *mindieLogToStdout = std::getenv("MINDIE_LOG_TO_STDOUT");
    if (mindieLogToStdout != nullptr) {
        LogUtils::SetMindieLogParamBool(loggerType, logToStdOut_, mindieLogToStdout);
        return;
    }
    std::string envToStdout;
    if (loggerType == LoggerType::ATB) {
        if (const char *env = std::getenv("ATB_LOG_TO_STDOUT")) {
            envToStdout = env;
        }
    } else if (loggerType == LoggerType::MINDIE_LLM) {
        if (const char *env = std::getenv("MINDIE_LLM_LOG_TO_STDOUT")) {
            envToStdout = env;
        }
    }
    // Avoid race conditions:
    if (!envToStdout.empty()) {
        logToStdOut_ = (envToStdout == "1");
        return;
    }
}

void LogConfig::InitLogToFileFlag(LoggerType loggerType)
{
    const char *mindieLogToFile = std::getenv("MINDIE_LOG_TO_FILE");
    if (mindieLogToFile != nullptr) {
        LogUtils::SetMindieLogParamBool(loggerType, logToFile_, mindieLogToFile);
        return;
    }
    std::string envToFile;
    if (loggerType == LoggerType::ATB) {
        if (const char *env = std::getenv("ATB_LOG_TO_FILE")) {
            envToFile = env;
        }
    } else if (loggerType == LoggerType::MINDIE_LLM) {
        if (const char *env = std::getenv("MINDIE_LLM_LOG_TO_FILE")) {
            envToFile = env;
        }
    }
    // Avoid race conditions:
    if (!envToFile.empty()) {
        logToFile_ = (envToFile == "1");
        return;
    }
}

void LogConfig::InitLogLevel(LoggerType loggerType)
{
    const char *mindieLogLevel = std::getenv("MINDIE_LOG_LEVEL");
    if (mindieLogLevel != nullptr) {
        LogUtils::SetMindieLogParamLevel(loggerType, logLevel_, mindieLogLevel);
        return;
    }
    std::string envLevel;
    if (loggerType == LoggerType::ATB) {
        if (const char *env = std::getenv("ATB_LOG_LEVEL")) {
            envLevel = env;
        }
    } else if (loggerType == LoggerType::MINDIE_LLM) {
        if (const char *env = std::getenv("MINDIE_LLM_LOG_LEVEL")) {
            envLevel = env;
        }
    }
    if (!envLevel.empty()) {
        std::transform(envLevel.begin(), envLevel.end(), envLevel.begin(), ::toupper);
        auto iter = LOG_LEVEL_MAP.find(envLevel);
        if (iter != LOG_LEVEL_MAP.end()) {
            logLevel_ = iter->second;
            return;
        }
    }
    logLevel_ = DEFAULT_LOG_LEVEL;
}

void LogConfig::InitLogFilePath(LoggerType loggerType)
{
    const char *mindieLogPath = std::getenv("MINDIE_LOG_PATH");
    std::string lastDir = "/debug";
    if (loggerType == LoggerType::SECURITY) {
        lastDir = "/security";
    }
    if (mindieLogPath != nullptr) {
        LogUtils::SetMindieLogParamString(loggerType, logFilePath_, mindieLogPath);
        if (logFilePath_[0] != '/') {
            logFilePath_ = DEFAULT_LOG_PATH + "/" + logFilePath_ + lastDir;
        } else {
            logFilePath_ = logFilePath_ + "/log" + lastDir;
        }
    } else {
        logFilePath_ = DEFAULT_LOG_PATH + lastDir;
    }
    LogUtils::GetLogFileName(loggerType, logFilePath_);
}

void LogConfig::InitLogVerbose(LoggerType loggerType)
{
    const char *mindieLogVerbose = std::getenv("MINDIE_LOG_VERBOSE");
    if (mindieLogVerbose != nullptr) {
        LogUtils::SetMindieLogParamBool(loggerType, logVerbose_, mindieLogVerbose);
        return;
    }
}

void LogConfig::InitLogRotationParam(LoggerType loggerType)
{
    const char *mindieLogRotate = std::getenv("MINDIE_LOG_ROTATE");
    if (mindieLogRotate != nullptr) {
        LogUtils::SetMindieLogParamString(loggerType, logRotateConfig_, mindieLogRotate);
        LogUtils::UpdateLogFileParam(logRotateConfig_, logFileSize_, logFileCount_);
        return;
    }
}

void CheckLogEnv()
{
    std::unordered_map<std::string, std::string> deprecatedEnv = {
        {"OCK_LOG_TO_STDOUT", "MINDIE_LOG_TO_STDOUT"},
        {"OCK_LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"MINDIE_LLM_PYTHON_LOG_TO_FILE", "MINDIE_LOG_TO_FILE"},
        {"MINDIE_LLM_PYTHON_LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"MINDIE_LLM_PYTHON_LOG_PATH", "MINDIE_LOG_PATH"},
        {"MINDIE_LLM_PYTHON_LOG_TO_STDOUT", "MINDIE_LOG_TO_STDOUT"},
        {"MINDIE_LLM_LOG_TO_FILE", "MINDIE_LOG_TO_FILE"},
        {"MINDIE_LLM_LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"MINDIE_LLM_LOG_TO_STDOUT", "MINDIE_LOG_TO_STDOUT"},
        {"ATB_LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"ATB_LOG_TO_FILE", "MINDIE_LOG_TO_FILE"},
        {"ATB_LOG_TO_STDOUT", "MINDIE_LOG_TO_STDOUT"},
        {"LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"LOG_TO_FILE", "MINDIE_LOG_TO_FILE"},
        {"MINDIE_LLM_PYTHON_LOG_MAXNUM", "MINDIE_LOG_ROTATE"},
        {"MINDIE_LLM_PYTHON_LOG_MAXSIZE", "MINDIE_LOG_ROTATE"}};
    for (auto it = deprecatedEnv.begin(); it != deprecatedEnv.end(); ++it) {
        const char *env = std::getenv(it->first.c_str());
        if (env != nullptr) {
            std::cout << "The old environment variable " << it->first
                      << " will be deprecated on 2025/12/31. Please use the new environment variable " << it->second
                      << " as soon as possible." << std::endl;
        }
    }
}

int LogConfig::ValidateSettings()
{
    if (!logToFile_) {
        return LOG_OK;
    }
    if (!CheckAndGetLogPath(logFilePath_)) {
        std::cout << "Cannot get the log path." << std::endl;
        return LOG_INVALID_PARAM;
    }
    if (logFileSize_ > MAX_ROTATION_FILE_SIZE_LIMIT || logFileSize_ < MIN_ROTATION_FILE_SIZE_LIMIT) {
        std::cout << "Invalid max file size, which should be greater than " << MIN_ROTATION_FILE_SIZE_LIMIT
                  << " bytes and less than " << MAX_ROTATION_FILE_SIZE_LIMIT << " bytes." << std::endl;
        return LOG_INVALID_PARAM;
    }
    if (logFileCount_ > MAX_ROTATION_FILE_COUNT_LIMIT || logFileCount_ < MIN_ROTATION_FILE_COUNT_LIMIT) {
        std::cout << "Invalid max file count, which should be greater than " << MIN_ROTATION_FILE_COUNT_LIMIT
                  << " and less than " << MAX_ROTATION_FILE_COUNT_LIMIT;
        return LOG_INVALID_PARAM;
    }
    return LOG_OK;
}

void LogConfig::MakeDirsWithTimeOut(const std::string &parentPath) const
{
    uint32_t limitTime = 500;
    auto start = std::chrono::steady_clock::now();
    std::chrono::milliseconds timeout(limitTime);
    while (!FileSystem::Exists(parentPath)) {
        auto it = FileSystem::Makedirs(parentPath, MAX_LOG_DIR_PERM);
        if (it) {
            break;
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            std::cout << "Create dirs failed : timed out!" << std::endl;
            break;
        }
    }
}

bool LogConfig::CheckAndGetLogPath(const std::string &configLogPath)
{
    if (configLogPath.empty()) {
        std::cout << "The path of log in config is empty." << std::endl;
        return false;
    }

    std::string filePath = configLogPath;
    std::string baseDir = "/";
    if (configLogPath[0] != '/') {  // The configLogPath is relative.
        const char *homePath = std::getenv("HOME");
        if (homePath == nullptr) {
            throw std::runtime_error("homePath is null, please check env HOME");
        }
        baseDir = std::string(homePath);
        filePath = std::string(homePath) + "/" + configLogPath;
    }

    if (filePath.length() > MAX_PATH_LENGTH) {
        std::cout << "The path of log is too long: " << filePath << std::endl;
        return false;
    }
    size_t lastSlash = filePath.rfind('/', filePath.size() - 1);
    if (lastSlash == std::string::npos) {
        std::cout << "The form of logPath is invalid: " << filePath << std::endl;
        return false;
    }

    std::string parentPath = filePath.substr(0, lastSlash);
    std::string errMsg;

    MakeDirsWithTimeOut(parentPath);

    FileValidationParams params1 = {true, MAX_LOG_DIR_PERM, MAX_ROTATION_FILE_SIZE_LIMIT, true};
    if (!FileUtils::IsFileValid(parentPath.c_str(), errMsg, params1)) {
        throw std::runtime_error(errMsg);
    }

    if (filePath.size() > PATH_MAX) {
        std::cerr << "Error: Allowed maximum path length is " << PATH_MAX
            << ", but got " << filePath.size() << ". You can reduce log directory length.\n";
        throw std::runtime_error(std::string("Log file path length exceeds PATH_MAX."));
    }
    baseDir_ = baseDir;
    logFilePath_ = filePath;
    return true;
}

}  // namespace mindie_llm
