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
#include <ctime>
#include <memory>
#include <cstdio>
#include <iostream>
#include <sstream>
#include "infer_instances.h"
#include "health_checker.h"

namespace mindie_llm {

constexpr size_t EXECUTE_COMMAND_BUFFER_SIZE = 128;
constexpr size_t SHORT_BITS_SIZE = 3;


HealthChecker::HealthChecker() : mRunning(false)
{
    auto &configManager = mindie_llm::ConfigManager::GetInstance();
    mEngineName = configManager.GetBackendConfig().backendName;
    mServiceStatus.store(SERVICE_INIT);
    mEndpointStatusCode.store(STATUS_CODE_INIT);
    GetChipPerCard();
    statusTransferMap = {
        {STATUS_CODE_INIT, {STATUS_CODE_NORMAL}},
        {STATUS_CODE_NORMAL, {STATUS_CODE_PAUSE, STATUS_CODE_ABNORMAL}},
        {STATUS_CODE_PAUSE, {STATUS_CODE_READY, STATUS_CODE_ABNORMAL_PAUSE}},
        {STATUS_CODE_ABNORMAL, {STATUS_CODE_NORMAL, STATUS_CODE_ABNORMAL_PAUSE}},
        {STATUS_CODE_READY, {STATUS_CODE_NORMAL, STATUS_CODE_ABNORMAL_READY}},
        {STATUS_CODE_ABNORMAL_PAUSE, {STATUS_CODE_PAUSE, STATUS_CODE_ABNORMAL_READY}},
        {STATUS_CODE_ABNORMAL_READY, {STATUS_CODE_ABNORMAL, STATUS_CODE_READY}}
        // other transfers are invalid
    };
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Healthchecker instance created.");
}

void HealthChecker::GetChipPerCard()
{
    std::string cmd = "npu-smi info -t usages -i 0 | awk '/Chip Count/ {print $NF}'";
    try {
        std::string output = ExecuteCommand(cmd);
        mChipPerCard = std::stoi(output);
        if (mChipPerCard <= 0) {
            ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                      GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                      "HealthChecker: Invalid Chip Count value from npu-smi: " << output << ". Defaulting to 1.");
            mChipPerCard = 1;
        } else {
            ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Detected Chip Count: " << mChipPerCard);
        }
    } catch (const std::exception &e) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                  "HealthChecker: Failed to parse Chip Count from npu-smi output. Exception: " << e.what()
                                                                                               << ". Defaulting to 1.");
        mChipPerCard = 1;
    }
}

void HealthChecker::PrintNpuDeviceIds()
{
    std::shared_lock<std::shared_mutex> lock(mNpuDevicesMutex);
    std::stringstream ss;
    ss << "{";
    for (const auto &id : mNpuDeviceCardIds) {
        if (id != *mNpuDeviceCardIds.begin()) {
            ss << ", ";
        }
        ss << id;
    }
    ss << "}";
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: NPU Device Card IDs: " << ss.str());
}

std::string HealthChecker::StatusToString(const ServiceStatus &status) const
{
    switch (status) {
        case SERVICE_READY: return "SERVICE_READY";
        case SERVICE_NORMAL: return "SERVICE_NORMAL";
        case SERVICE_ABNORMAL: return "SERVICE_ABNORMAL";
        case SERVICE_PAUSE: return "SERVICE_PAUSE";
        case SERVICE_INIT: return "SERVICE_INIT";
        default: return "UNKNOWN";
    }
}

std::string HealthChecker::CodeToString(const EndpointStatusCode &code) const
{
    switch (code) {
        case STATUS_CODE_INIT: return "STATUS_CODE_INIT";
        case STATUS_CODE_NORMAL: return "STATUS_CODE_NORMAL";
        case STATUS_CODE_PAUSE: return "STATUS_CODE_PAUSE";
        case STATUS_CODE_ABNORMAL: return "STATUS_CODE_ABNORMAL";
        case STATUS_CODE_READY: return "STATUS_CODE_READY";
        case STATUS_CODE_ABNORMAL_PAUSE: return "STATUS_CODE_ABNORMAL_PAUSE";
        case STATUS_CODE_ABNORMAL_READY: return "STATUS_CODE_ABNORMAL_READY";
        default: return "UNKNOWN";
    }
}

HealthChecker::~HealthChecker()
{
    Stop();
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Healthchecker instance destroyed.");
}

bool HealthChecker::Start()
{
    if (!mRunning.load()) {
        mRunning.store(true);
        mCheckerThread = std::thread(&HealthChecker::CheckServiceStatus, this);
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Health check thread started.");
        return true;
    } else {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, STATUS_WARNING),
                  "HealthChecker: Attempted to start already running health check thread");
        return false;
    }
}

void HealthChecker::Stop()
{
    if (mRunning.load()) {
        mRunning.store(true);
        if (mCheckerThread.joinable()) {
            mCheckerThread.join();
        }
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Health check thread stopped.");
    } else {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, STATUS_WARNING),
                  "HealthChecker: Attempted to stop non-running health check thread");
    }
}

HealthChecker &HealthChecker::GetInstance()
{
    static HealthChecker instance;
    return instance;
}

ServiceStatus HealthChecker::GetServiceStatus() { return mServiceStatus.load(); }

EndpointStatusCode HealthChecker::GetEndpointStatusCode() { return mEndpointStatusCode.load(); }

bool HealthChecker::CheckErrorListEmpty() { return mErrorList.Empty(); }

void HealthChecker::GetStatusAndErrorList(ServiceStatus &status, std::vector<ErrorItem> &errorList)
{
    status = mServiceStatus.load();
    mErrorList.ForEach([&errorList](const ErrorItem &item) { errorList.push_back(item); }, mErrorList.Size());
    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: GetStatusAndErrorList called. Status: "
                                                << status << ", ErrorList size: " << errorList.size());
    mErrorList.Clear();
}

bool HealthChecker::CheckModelInstanceStarted() const
{
    bool isStarted = false;
    auto status = GetInferInstance()->CheckInferInstanceStarted(isStarted);
    if (!status.IsOk()) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                  "HealthChecker: Failed to get model instance started status. " << status.StatusMsg());
        return false;
    }
    return isStarted;
}

void HealthChecker::CheckServiceStatus()
{
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
              "HealthChecker: Starting health check loop with interval: " << checkIntervalSeconds << " seconds");
    while (mRunning.load()) {
        if (GetEndpointStatusCode() == STATUS_CODE_INIT && !CheckModelInstanceStarted()) {
            // Service not init
            ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Model instance not started, skipping check");
            std::this_thread::sleep_for(std::chrono::seconds(std::chrono::seconds(checkIntervalSeconds)));
            continue;
        } else if (!CheckErrorListEmpty()) {
            std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
            continue;
        } else if (!CheckAllNpuAicoreUsage() && !CheckVirtualInfer()) {
            // Service check abnormal
            if (GetEndpointStatusCode() == STATUS_CODE_PAUSE || GetEndpointStatusCode() == STATUS_CODE_READY) {
                // Health check ignored within qingqu linkdown recover process
                std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
                continue;
            }
            ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Service check abnormal, update status by ErrorItem");
            EnqueueErrorMessage(GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                                SUBMODLE_NAME_HEALTHCHECKER);
        } else {
            if (GetEndpointStatusCode() == 0b111) {
                ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                          "HealthChecker: Service check normal, update status from init to normal");
                UpdateStatusByCode(STATUS_CODE_NORMAL);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
    }
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Exiting health check loop.");
}

std::string HealthChecker::ExecuteCommand(const std::string &cmd) const
{
    std::array<char, EXECUTE_COMMAND_BUFFER_SIZE> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealthChecker: popen() failed!");
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::vector<int> HealthChecker::ParseAicoreUsage(const std::string &output) const
{
    std::vector<int> usages;
    std::istringstream iss(output);
    std::string line;
    int minUsagePercent = 0;
    int maxUsagePercent = 100;

    while (std::getline(iss, line)) {
        try {
            float usagePercent = std::stof(line);
            if (usagePercent < minUsagePercent || usagePercent > maxUsagePercent) {
                ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                          GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                          "HealthChecker: Invalid Aicore usage percentage" << usagePercent);
                usages.push_back(0);
            } else {
                usages.push_back(usagePercent);
            }
        } catch (...) {
            ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                       GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                       "HealthChecker: Failed to parse Aicore usage value: " << line);
        }
    }

    if (static_cast<int>(usages.size()) != mChipPerCard) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                  "HealthChecker: The number of Aicore usage " << usages.size() << " is not equal to the chip count "
                                                               << mChipPerCard);
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR), output);
    }

    return usages;
}

std::vector<int> HealthChecker::GetAicoreUsages(int npuId) const
{
    try {
        std::ostringstream cmdStream;
        cmdStream << "npu-smi info -i " << npuId << " -t usages | awk '/Aicore Usage/ {print $NF}'";
        std::string cmd = cmdStream.str();
        std::string output = ExecuteCommand(cmd);
        if (output.empty()) {
            ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                       GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                       "HealchChecker: Empty output from npu-smi for npu " << npuId);
            return {};
        } else {
            return ParseAicoreUsage(output);
        }
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealchChecker: Failed to get aicore of npu " << npuId);
        return {};
    }
}

bool HealthChecker::CheckAllNpuAicoreUsage()
{
    std::set<int> npuIdsCopy;
    {
        std::shared_lock<std::shared_mutex> lock(mNpuDevicesMutex);
        npuIdsCopy = mNpuDeviceCardIds;
    }
    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER,
               "HealthChecker: Checking NPU Aicore usage for " << npuIdsCopy.size() << " devices");
    if (npuIdsCopy.empty()) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                  "HealchChecker: No npu devices configured");
        return false;
    }

    bool allHealthy = true;
    int minUsagePercent = 10;
    for (const auto &npuId : npuIdsCopy) {
        try {
            std::vector<int> chip_usages = GetAicoreUsages(npuId);
            if (chip_usages.empty()) {
                allHealthy = false;
                ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: NPU Aicore usage is empty.");
            } else {
                for (const auto &usage : chip_usages) {
                    if (usage < minUsagePercent) {
                        allHealthy = false;
                        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                                  "HealthChecker: NPU " << npuId << " Aicore usage < 10%.");
                        break;
                    }
                }
            }
        } catch (const std::exception &e) {
            ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                       GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
                       "HealthChecker: Exception when checking NPU " << npuId << ": " << e.what());
            allHealthy = false;
        }
    }
    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER,
               "HealthChecker: Overall NPU health status: " << (allHealthy ? "normal." : "abnormal."));
    return allHealthy;
}

bool HealthChecker::CheckVirtualInfer() const
{
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Starting virtual inference check.");
    return true;
}

bool HealthChecker::IsValidStatusTransition(const EndpointStatusCode &from, const EndpointStatusCode &to)
{
    if (statusTransferMap.find(from) == statusTransferMap.end() ||
        std::find(statusTransferMap[from].begin(), statusTransferMap[from].end(), to) ==
        statusTransferMap[from].end()) {
        return false;
    }
    return true;
}

void HealthChecker::UpdateStatusByCode(const EndpointStatusCode &code)
{
    if (mEndpointStatusCode.load() == code) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Status unchanged: " << StatusToString(mServiceStatus));
        return;
    }
    if (!IsValidStatusTransition(mEndpointStatusCode.load(), code)) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealthChecker: Invalid status transition from " << CodeToString(mEndpointStatusCode.load())
                                                                      << " to " << CodeToString(code));
        return;
    }

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Status code changed from "
                                               << CodeToString(mEndpointStatusCode.load()) << " to "
                                               << CodeToString(code));
    ServiceStatus newStatus;
    if ((code & 0b001) != 0) {
        newStatus = SERVICE_ABNORMAL;
    } else if ((code & 0b010) != 0) {
        newStatus = SERVICE_READY;
    } else if ((code & 0b100) != 0) {
        newStatus = SERVICE_PAUSE;
    } else {
        newStatus = SERVICE_NORMAL;
    }
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Service status changed from "
                                               << StatusToString(mServiceStatus) << " to "
                                               << StatusToString(newStatus));
    mServiceStatus.store(newStatus);
    mEndpointStatusCode.store(code);
}

void HealthChecker::EnqueueErrorMessage(const std::string &errCode, const std::string &createdBy,
                                        const std::string &deviceIP, const int &deviceID,
                                        const std::chrono::time_point<std::chrono::system_clock> &timestamp)
{
    ErrorItem item(errCode, createdBy, deviceIP, deviceID, timestamp);

    if (mErrorList.Size() >= maxErrorListSize) {
        ErrorItem itemToRemove;
        mErrorList.PopFront(itemToRemove);
    }
    mErrorList.PushBack(item);
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: New error added. Error code: "
                                               << errCode << ", createdBy: " << createdBy
                                               << ", deviceIP: " << deviceIP << ", deviceID: " << deviceID);
    if (mEndpointStatusCode.load() == STATUS_CODE_NORMAL) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                  "HealthChecker: Service status changed from SERVICE_NORAML to SERVICE_ABNORMAL newStatus");
        mServiceStatus.store(SERVICE_ABNORMAL);
        mEndpointStatusCode.store(STATUS_CODE_ABNORMAL);
    }
}

void HealthChecker::UpdateNpuDeviceIds(const std::set<int> &npuDeviceIds)
{
    {
        std::unique_lock<std::shared_mutex> lock(mNpuDevicesMutex);
        mNpuDeviceCardIds.clear();
        for (const auto &id : npuDeviceIds) {
            mNpuDeviceCardIds.insert(id / mChipPerCard);
        }
    }
    PrintNpuDeviceIds();
}

} // namespace mindie_llm